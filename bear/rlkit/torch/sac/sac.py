from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer
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
            # unc_critic = torch.clamp(unc, 0.0, 1.5)
            unc_critic = unc
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


class SACTrainer(TorchTrainer):
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

            discount=0.99,
            reward_scale=1.0,

            policy_lr=1e-3,
            qf_lr=1e-3,
            optimizer_class=optim.Adam,

            soft_target_tau=1e-2,
            target_update_period=1,
            plotter=None,
            render_eval_paths=False,

            use_automatic_entropy_tuning=True,
            target_entropy=None,
            beta = 1
    ):
        super().__init__()
        self.env = env
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period

        self.T = 100
        self.beta = beta

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(self.env.action_space.shape).item()  # heuristic value from Tuomas
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )

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

        self.discount = discount
        self.reward_scale = reward_scale
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

        self.discrete = False
        self.pre_model_name = pre_model
        self.pre_model = unc_premodel(self.env, env_name, pre_model)

    def train_from_torch(self, batch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        # print uncertainty

        critic_unc = uncertainty(obs, actions, self.T, self.beta, self.pre_model, self.pre_model_name)

        print(critic_unc)

        exit()

        """
        Policy and Alpha Loss
        """
        new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(
            obs, reparameterize=True, return_log_prob=True,
        )
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = 1

        q_new_actions = torch.min(
            self.qf1(obs, new_obs_actions),
            self.qf2(obs, new_obs_actions),
        )
        policy_loss = (alpha*log_pi - q_new_actions).mean()

        """
        QF Loss
        """
        q1_pred = self.qf1(obs, actions)
        q2_pred = self.qf2(obs, actions)
        # Make sure policy accounts for squashing functions like tanh correctly!
        new_next_actions, _, _, new_log_pi, *_ = self.policy(
            next_obs, reparameterize=True, return_log_prob=True,
        )
        target_q_values = torch.min(
            self.target_qf1(next_obs, new_next_actions),
            self.target_qf2(next_obs, new_next_actions),
        ) - alpha * new_log_pi

        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
        qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
        qf2_loss = self.qf_criterion(q2_pred, q_target.detach())

        """
        Update networks
        """
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        """
        Soft Updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(
                self.qf1, self.target_qf1, self.soft_target_tau
            )
            ptu.soft_update_from_to(
                self.qf2, self.target_qf2, self.soft_target_tau
            )

        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            policy_loss = (log_pi - q_new_actions).mean()

            self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(q2_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))
            if self.use_automatic_entropy_tuning:
                self.eval_statistics['Alpha'] = alpha.item()
                self.eval_statistics['Alpha Loss'] = alpha_loss.item()
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
        ]

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.qf1,
            target_qf2=self.qf2,
        )

