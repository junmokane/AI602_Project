from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn
import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer
from torch import autograd
from rlkit.torch.networks import FlattenMlp_Dropout
from uncertainty_modeling.rl_uncertainty.rank1.r1bnn import Model
from uncertainty_modeling.rl_uncertainty.model import RegNetBase, SWAG, RaPP, get_diffs


def unc_premodel(env, env_name, model_name):
    path = './uncertainty_modeling/rl_uncertainty'
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size
    input_size = obs_dim + action_dim
    model = None
    if model_name == 'mc_dropout':
        model = FlattenMlp_Dropout(  # Check the dropout layer!
            input_size=input_size,
            output_size=1,
            hidden_sizes=[256, 256],
        ).cuda()
    if model_name == 'rank1':
        model = Model(x_dim=input_size, h_dim=10, y_dim=1, n=10).cuda()
    if model_name == 'rapp':
        model = RaPP(input_size).cuda()
    if model_name == 'swag':
        kwargs = {"dimensions": [200, 50, 50, 50],
                  "output_dim": 1,
                  "input_dim": input_size}
        args = list()
        model = SWAG(RegNetBase, subspace_type="pca", *args, **kwargs,
                     subspace_kwargs={"max_rank": 10, "pca_rank": 10})
        model.cuda()
    if model == None:
        raise AttributeError
    else:
        model.load_state_dict(torch.load('{}/{}/model/{}/model_1980.pt'.format(path, model_name, env_name)))

        return model


def uncertainty(state, action, rep, beta, pre_model, pre_model_name):
    with torch.no_grad():
        batch_size = state.shape[0]

        if pre_model_name == 'rapp':
            dif = get_diffs(torch.cat([state, action], dim=1), pre_model)
            difs = torch.cat([torch.from_numpy(i) for i in dif], dim=-1).cuda()
            dif = (difs ** 2).mean(axis=1)
            unc = beta / dif  # B
            unc = unc.unsqueeze(1) # Bx1
            # TODO: clipping on uncertainty
            unc_critic = torch.clamp(unc, 0.0, 1.5)
            return unc_critic

        if pre_model_name == 'rank1':
           rep = rep // pre_model.n

        if pre_model_name == 'swag':
            temp = []
            for i in range(rep):
                pre_model.sample(scale=10.)
                target_qf1 = pre_model(torch.cat([state, action], dim=1))  # BTx1
                temp.append(target_qf1)
            target_qf1 = torch.cat(temp, dim=0)
        else:
            state_cp = state.unsqueeze(1).repeat(1, rep, 1).view(state.shape[0] * rep, state.shape[1])
            action_cp = action.unsqueeze(1).repeat(1, rep, 1).view(action.shape[0] * rep, action.shape[1])
            target_qf1 = pre_model(torch.cat([state_cp, action_cp], dim=1))  # BTx1

        if pre_model_name == 'rank1':
            rep = rep * pre_model.n
            target_qf1 = target_qf1.view(rep, batch_size, 1)  # BxTx1

            q_sq = torch.mean(target_qf1 ** 2, dim=0)  # Bx1
            q_mean_sq = torch.mean(target_qf1, dim=0) ** 2  # Bx1
        else:
            target_qf1 = target_qf1.view(batch_size, rep, 1)  # BxTx1

            q_sq = torch.mean(target_qf1 ** 2, dim=1)  # Bx1
            q_mean_sq = torch.mean(target_qf1, dim=1) ** 2  # Bx1

        # var = torch.std(target_qf1, dim=1)
        var = q_sq - q_mean_sq
        unc = beta / var  # Bx1
        # TODO: clipping on uncertainty
        unc_critic = torch.clamp(unc, 0.0, 1.5)
    return unc_critic


class MUSATTrainer(TorchTrainer):
    def __init__(
            self,
            pre_model,
            env_name,
            env,
            policy,
            qf1,
            qf2,
            target_qf1,
            target_qf2,
            vae,

            discount=0.99,
            reward_scale=1.0,

            policy_lr=1e-3,
            qf_lr=1e-3,
            optimizer_class=optim.Adam,

            soft_target_tau=1e-2,
            target_update_period=1,
            plotter=None,
            render_eval_paths=False,

            # BEAR specific params
            mode='auto',
            kernel_choice='laplacian',
            policy_update_style=0,
            mmd_sigma=10.0,
            target_mmd_thresh=0.05,
            num_samples_mmd_match=4,
            use_target_nets=True,
            beta=1.0,
    ):
        super().__init__()
        self.env = env
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.vae = vae
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        self.T = 100
        self.beta = beta

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )
        self.vae_optimizer = optimizer_class(
            self.vae.parameters(),
            lr=3e-4,
        )

        self.mode = mode
        if self.mode == 'auto':
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=1e-3,
            )
        self.mmd_sigma = mmd_sigma
        self.kernel_choice = kernel_choice
        self.num_samples_mmd_match = num_samples_mmd_match
        self.policy_update_style = policy_update_style
        self.target_mmd_thresh = target_mmd_thresh

        self.discount = discount
        self.reward_scale = reward_scale
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

        self.discrete = False
        self._current_epoch = 0
        self._policy_update_ctr = 0
        self._num_q_update_steps = 0
        self._num_policy_update_steps = 0
        self.pre_model = unc_premodel(self.env, env_name, pre_model)
        self.pre_model_name = pre_model

    def eval_q_custom(self, custom_policy, data_batch, q_function=None):
        if q_function is None:
            q_function = self.qf1

        obs = data_batch['observations']
        # Evaluate policy Loss
        new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(
            obs, reparameterize=True, return_log_prob=True,
        )
        q_new_actions = q_function(obs, new_obs_actions)
        return float(q_new_actions.mean().detach().cpu().numpy())

    def mmd_loss_laplacian(self, samples1, samples2, sigma=0.2):
        """MMD constraint with Laplacian kernel for support matching"""
        # sigma is set to 20.0 for hopper, cheetah and 50 for walker/ant
        diff_x_x = samples1.unsqueeze(2) - samples1.unsqueeze(1)  # B x N x N x d
        diff_x_x = torch.mean((-(diff_x_x.abs()).sum(-1) / (2.0 * sigma)).exp(), dim=(1, 2))

        diff_x_y = samples1.unsqueeze(2) - samples2.unsqueeze(1)
        diff_x_y = torch.mean((-(diff_x_y.abs()).sum(-1) / (2.0 * sigma)).exp(), dim=(1, 2))

        diff_y_y = samples2.unsqueeze(2) - samples2.unsqueeze(1)  # B x N x N x d
        diff_y_y = torch.mean((-(diff_y_y.abs()).sum(-1) / (2.0 * sigma)).exp(), dim=(1, 2))

        overall_loss = (diff_x_x + diff_y_y - 2.0 * diff_x_y + 1e-6).sqrt()
        return overall_loss

    def mmd_loss_gaussian(self, samples1, samples2, sigma=0.2):
        """MMD constraint with Gaussian Kernel support matching"""
        # sigma is set to 20.0 for hopper, cheetah and 50 for walker/ant
        diff_x_x = samples1.unsqueeze(2) - samples1.unsqueeze(1)  # B x N x N x d
        diff_x_x = torch.mean((-(diff_x_x.pow(2)).sum(-1) / (2.0 * sigma)).exp(), dim=(1, 2))

        diff_x_y = samples1.unsqueeze(2) - samples2.unsqueeze(1)
        diff_x_y = torch.mean((-(diff_x_y.pow(2)).sum(-1) / (2.0 * sigma)).exp(), dim=(1, 2))

        diff_y_y = samples2.unsqueeze(2) - samples2.unsqueeze(1)  # B x N x N x d
        diff_y_y = torch.mean((-(diff_y_y.pow(2)).sum(-1) / (2.0 * sigma)).exp(), dim=(1, 2))

        overall_loss = (diff_x_x + diff_y_y - 2.0 * diff_x_y + 1e-6).sqrt()
        return overall_loss

    def train_from_torch(self, batch):
        self._current_epoch += 1
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        """
        Behavior clone a policy
        """
        recon, mean, std = self.vae(obs, actions)
        recon_loss = self.qf_criterion(recon, actions)
        kl_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        vae_loss = recon_loss + 0.5 * kl_loss

        self.vae_optimizer.zero_grad()
        vae_loss.backward()
        self.vae_optimizer.step()

        """
        Critic Training
        """
        with torch.no_grad():
            # Duplicate state 10 times (10 is a hyperparameter chosen by BCQ)
            state_rep = next_obs.unsqueeze(1).repeat(1, 10, 1).view(next_obs.shape[0] * 10, next_obs.shape[1])  # 10BxS
            # Compute value of perturbed actions sampled from the VAE
            action_rep = self.policy(state_rep)[0]  # 10BxA
            target_qf1 = self.target_qf1(state_rep, action_rep)  # 10Bx1
            target_qf2 = self.target_qf2(state_rep, action_rep)  # 10Bx1
            # Soft Clipped Double Q-learning
            target_Q = 0.75 * torch.min(target_qf1, target_qf2) + 0.25 * torch.max(target_qf1, target_qf2)  # 10Bx1
            max_target_action = target_Q.view(next_obs.shape[0], -1).max(1)  # 10Bx1 > Bx10 > B,B
            target_Q = max_target_action[0].view(-1, 1)  # B > Bx1
            # 10BxA > Bx10xA > BxA
            max_actions = action_rep.view(next_obs.shape[0], 10, action_rep.shape[1])[torch.arange(next_obs.shape[0]),
                                                                                      max_target_action[1]]
            target_Q = self.reward_scale * rewards + (1.0 - terminals) * self.discount * target_Q  # Bx1

        qf1_pred = self.qf1(obs, actions)  # Bx1
        qf2_pred = self.qf2(obs, actions)  # Bx1
        critic_unc = uncertainty(next_obs, max_actions, self.T, self.beta, self.pre_model, self.pre_model_name)
        qf1_loss = ((qf1_pred - target_Q.detach()).pow(2) * critic_unc).mean()
        qf2_loss = ((qf2_pred - target_Q.detach()).pow(2) * critic_unc).mean()


        """
        Actor Training
        """
        sampled_actions, raw_sampled_actions = self.vae.decode_multiple(obs, num_decode=self.num_samples_mmd_match)
        actor_samples, _, _, _, _, _, _, raw_actor_actions = self.policy(
            obs.unsqueeze(1).repeat(1, self.num_samples_mmd_match, 1).view(-1, obs.shape[1]), return_log_prob=True)
        actor_samples = actor_samples.view(obs.shape[0], self.num_samples_mmd_match, actions.shape[1])
        raw_actor_actions = raw_actor_actions.view(obs.shape[0], self.num_samples_mmd_match, actions.shape[1])

        if self.kernel_choice == 'laplacian':
            mmd_loss = self.mmd_loss_laplacian(raw_sampled_actions, raw_actor_actions, sigma=self.mmd_sigma)
        elif self.kernel_choice == 'gaussian':
            mmd_loss = self.mmd_loss_gaussian(raw_sampled_actions, raw_actor_actions, sigma=self.mmd_sigma)

        action_divergence = ((sampled_actions - actor_samples) ** 2).sum(-1)
        raw_action_divergence = ((raw_sampled_actions - raw_actor_actions) ** 2).sum(-1)

        q_val1 = self.qf1(obs, actor_samples[:, 0, :])
        q_val2 = self.qf2(obs, actor_samples[:, 0, :])
        actor_unc = uncertainty(obs, actor_samples[:, 0, :], self.T, self.beta, self.pre_model, self.pre_model_name)

        if self.policy_update_style == '0':
            policy_loss = torch.min(q_val1, q_val2)[:, 0] * actor_unc[:, 0]
        elif self.policy_update_style == '1':
            policy_loss = torch.mean(q_val1, q_val2)[:, 0] * actor_unc[:, 0]

        # Use uncertainty after some epochs
        if self._n_train_steps_total >= 40000:
            if self.mode == 'auto':
                policy_loss = (-policy_loss + self.log_alpha.exp() * (mmd_loss - self.target_mmd_thresh)).mean()
            else:
                policy_loss = (-policy_loss + 100 * mmd_loss).mean()
        else:
            if self.mode == 'auto':
                policy_loss = (self.log_alpha.exp() * (mmd_loss - self.target_mmd_thresh)).mean()
            else:
                policy_loss = 100 * mmd_loss.mean()

        """
        Update Networks
        """
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()

        self.policy_optimizer.zero_grad()
        if self.mode == 'auto':
            policy_loss.backward(retain_graph=True)
        self.policy_optimizer.step()

        if self.mode == 'auto':
            self.alpha_optimizer.zero_grad()
            (-policy_loss).backward()
            self.alpha_optimizer.step()
            self.log_alpha.data.clamp_(min=-5.0, max=10.0)

        """
        Update networks
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(
                self.qf1, self.target_qf1, self.soft_target_tau
            )
            ptu.soft_update_from_to(
                self.qf2, self.target_qf2, self.soft_target_tau
            )

        """
        Some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics['Num Q Updates'] = self._num_q_update_steps
            self.eval_statistics['Num Policy Updates'] = self._num_policy_update_steps
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(qf1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(qf2_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(target_Q),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'MMD Loss',
                ptu.get_numpy(mmd_loss)
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Action Divergence',
                ptu.get_numpy(action_divergence)
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Raw Action Divergence',
                ptu.get_numpy(raw_action_divergence)
            ))
            if self.mode == 'auto':
                self.eval_statistics['Alpha'] = self.log_alpha.exp().item()

        self._n_train_steps_total += 1

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
            self.vae
        ]

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.target_qf1,
            target_qf2=self.target_qf2,
            vae=self.vae,
        )