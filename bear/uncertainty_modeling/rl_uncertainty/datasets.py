import numpy as np
from torch.utils.data import Dataset

halfcheetah_task_name = ['halfcheetah-random-v0',
                         'halfcheetah-medium-v0',
                         'halfcheetah-expert-v0',
                         'halfcheetah-medium-replay-v0',
                         'halfcheetah-medium-expert-v0']

hopper_task_name = ['hopper-random-v0',
'hopper-medium-v0',
'hopper-expert-v0',
'hopper-medium-replay-v0',
'hopper-medium-expert-v0'
                 ]

walker_task_name = ['walker2d-random-v0',
'walker2d-medium-v0',
'walker2d-expert-v0',
'walker2d-medium-replay-v0',
'walker2d-medium-expert-v0'
                 ]


class GymDataset(Dataset):
    def __init__(self, env):
        dataset = env.get_dataset()
        all_obs = np.array(dataset['observations'])

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

