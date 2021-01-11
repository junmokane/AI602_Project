import argparse
import gym
import d4rl
import numpy as np

from rlkit.envs import ENVS
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.sac_uncertainty import SACTrainer
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from rlkit.envs.read_hdf5 import get_dataset


def load_hdf5(dataset, replay_buffer, max_size):
    all_obs = dataset['observations']
    all_act = dataset['actions']
    N = min(all_obs.shape[0], max_size)

    _obs = all_obs[:N-1]
    _actions = all_act[:N-1]
    _next_obs = all_obs[1:]
    _rew = np.squeeze(dataset['rewards'][:N-1])
    _rew = np.expand_dims(np.squeeze(_rew), axis=-1)
    _done = np.squeeze(dataset['terminals'][:N-1])
    _done = (np.expand_dims(np.squeeze(_done), axis=-1)).astype(np.int32)

    max_length = 100
    ctr = 0
    ## Only for MuJoCo environments
    ## Handle the condition when terminal is not True and trajectory ends due to a timeout
    for idx in range(_obs.shape[0]):
        if ctr >= max_length - 1:
            ctr = 0
        else:
            replay_buffer.add_sample_only(_obs[idx], _actions[idx], _rew[idx], _next_obs[idx], _done[idx])
            ctr += 1
            if _done[idx][0]:
                ctr = 0


def experiment(variant):
    env_name = variant['env_name']
    if env_name in ENVS:
        eval_env = NormalizedBoxEnv(ENVS[env_name]())
        expl_env = eval_env
    else:
        eval_env = NormalizedBoxEnv(gym.make(variant['env_name']))
        expl_env = eval_env

    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    M = variant['layer_size']
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
    )
    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        policy,
    )
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )

    file_path = './data/sac-point-robot/2021_01_04_22_25_16_exp_0000_s_0/offline_buffer_itr_140.hdf5'
    load_hdf5(get_dataset(file_path), replay_buffer, max_size=variant['replay_buffer_size'])

    trainer = SACTrainer(
        pre_model=args.pre_model,
        env_name=args.env,
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs']
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        batch_rl=True,  # this is for offline RL (False if online)
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()




if __name__ == "__main__":
    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser(description='Offline_SAC_Unc-runs')
    parser.add_argument("--env", type=str, default='halfcheetah-random-v0')
    parser.add_argument("--gpu", default='0', type=str)
    parser.add_argument('--qf_lr', default=3e-4, type=float)
    parser.add_argument('--policy_lr', default=1e-4, type=float)
    parser.add_argument('--seed', default=0, type=int)
    # training specs YOU SHOULD CONSIDER THIS!
    parser.add_argument("--max_path_length", type=int, default=1000)
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--num_eval_steps_per_epoch", type=int, default=5000)
    parser.add_argument("--num_trains_per_train_loop", type=int, default=1000)
    parser.add_argument("--num_expl_steps_per_train_loop", type=int, default=1000)
    parser.add_argument("--min_num_steps_before_training", type=int, default=1000)
    parser.add_argument('--pre_model', default='rapp', type=str)
    parser.add_argument('--policy_eval_start', type=int, default=40000)
    parser.add_argument('--beta', type=float, default=1.0)
    args = parser.parse_args()

    variant = dict(
        algorithm="Offline_SAC_Unc",
        version="normal",
        layer_size=256,
        replay_buffer_size=int(1E6),
        env_name=args.env,
        algorithm_kwargs=dict(
            num_epochs=args.num_epochs,
            num_eval_steps_per_epoch=args.num_eval_steps_per_epoch,
            num_trains_per_train_loop=args.num_trains_per_train_loop,
            num_expl_steps_per_train_loop=args.num_expl_steps_per_train_loop,
            min_num_steps_before_training=args.min_num_steps_before_training,
            max_path_length=args.max_path_length,
            batch_size=256,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
            policy_eval_start=args.policy_eval_start,  # 4000 for point-robot, 40000 for mujoco
            beta=args.beta,
        ),
    )
    setup_logger(exp_prefix='sac-' + args.env,
                 variant=variant,
                 text_log_file="debug.log",
                 variant_log_file="variant.json",
                 tabular_log_file="progress.csv",
                 snapshot_mode="gap_and_last",
                 snapshot_gap=5,
                 log_tabular_only=False,
                 log_dir=None,
                 git_infos=None,
                 script_name=None,
                 # **create_log_dir_kwargs
                 base_log_dir='./data',
                 exp_id=2,
                 seed=args.seed)  # if want to specify something more
    ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    experiment(variant)
