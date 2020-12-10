import numpy as np
from torch.utils.data import Dataset
import torch
# from rlkit.envs.half_cheetah import HalfCheetahEnv
from gym.envs.mujoco import HalfCheetahEnv as HalfCheetahEnv_
import gym
import d4rl

class ScatterDataset(Dataset):
    def __init__(self, path):
        self.datas = np.load(path)
        self.X, self.Y = self.datas[:, 0], self.datas[:, 1]

    def __getitem__(self, index):
        x = np.array(self.X[index % len(self.X)])
        y = np.array(self.Y[index % len(self.Y)])
        w = torch.rand([64])

        return {'input' : x, 'label' : y, 'weight' : w}

    def __len__(self):
        return len(self.X)

halfcheetah_task_name = ['halfcheetah-random-v0',
                         'halfcheetah-medium-v0',
                         'halfcheetah-expert-v0',
                         'halfcheetah-medium-replay-v0',
                         'halfcheetah-medium-expert-v0']

ant_task_name = ['ant-medium-expert-v0',
                 'ant-random-expert-v0',
                 'ant-medium-replay-v0',
                 'ant-medium-v0',
                 'ant-random-v0',
                 'ant-expert-v0',
                 ]

walker_task_name = ['walker2d-random-v0',
'walker2d-medium-v0',
'walker2d-expert-v0',
'walker2d-medium-replay-v0',
'walker2d-medium-expert-v0'
                 ]

class GymDataset(Dataset):
    def __init__(self):
        i = 0
        env = gym.make(walker_task_name[i])

        dataset = env.get_dataset()
        all_obs = np.array(dataset['observations'])

        # # Normalization
        # all_obs = 2*(all_obs - np.min(all_obs)) / (np.max(all_obs) - np.min(all_obs)) - 1

        all_act = np.array(dataset['actions'])
        N = all_obs.shape[0]

        _obs = all_obs[:N - 1]
        _actions = all_act[:N - 1]
        _next_obs = all_obs[1:]
        _next_actions = all_act[1:]
        _rew = np.squeeze(dataset['rewards'][:N - 1])
        _rew = np.expand_dims(np.squeeze(_rew), axis=-1)
        _done = np.squeeze(dataset['terminals'][:N - 1])
        _done = (np.expand_dims(np.squeeze(_done), axis=-1)).astype(np.int32)

        # reward = np.array(dataset['rewards'])

        # # Normalization
        # _rew = 2 * (_rew - np.min(_rew)) / (np.max(_rew) - np.min(_rew)) - 1

        self.obs_act = np.concatenate([_obs, _actions], axis=1)
        self.next_obs_act = np.concatenate([_next_obs, _next_actions], axis=1)
        self.rewards = _rew
        self.terminals = _done

    def __getitem__(self, index):
        obs_act = self.obs_act[index % len(self.obs_act)]
        next_obs_act = self.next_obs_act[index % len(self.obs_act)]
        rewards = self.rewards[index % len(self.obs_act)]
        terminals = self.terminals[index % len(self.obs_act)]

        return {'obs_act' : obs_act, 'next_obs_act' : next_obs_act, 'rewards' : rewards, 'terminals' : terminals}

    def __len__(self):
        return len(self.obs_act)

