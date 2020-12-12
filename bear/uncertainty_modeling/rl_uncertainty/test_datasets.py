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
    def __init__(self, env, ood_test, env_name):
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

        if ood_test == True:
            ood_action_idx = []
            id_action_idx = []
            if 'cheetah' in env_name:
                for i, action in enumerate(_actions):
                    if sum(action > 0.5) > 3:
                        ood_action_idx.append(i)
                    else:
                        id_action_idx.append(i)
            elif 'hopper' in env_name:
                for i, action in enumerate(_actions):
                    if sum(action > 0) > 2:
                        ood_action_idx.append(i)
                    else:
                        id_action_idx.append(i)

            elif 'walker' in env_name:
                print(np.max(_actions))
                print(np.min(_actions))
                for i, action in enumerate(_actions):
                    if sum(action > 0.6) > 3:
                        ood_action_idx.append(i)
                    else:
                        id_action_idx.append(i)
                print(len(id_action_idx))
                print(len(ood_action_idx))
            else:
                raise TypeError


            id_obs = _obs[id_action_idx]
            id_action = _actions[id_action_idx]
            ood_obs = _obs[ood_action_idx]
            ood_action = _actions[ood_action_idx]

            self.id_obs_act = np.concatenate([id_obs, id_action], axis=1)
            self.ood_obs_act = np.concatenate([ood_obs, ood_action], axis=1)

        else:
            id_obs = np.repeat(_obs, 2, axis = 0)
            id_action = np.repeat(_actions, 2, axis = 0)
            ood_obs = np.repeat(_obs, 2, axis = 0)
            ood_action = np.concatenate([np.ones_like(_actions), -np.ones_like(_actions)], axis=0)

            self.id_obs_act = np.concatenate([id_obs, id_action], axis=1)
            self.ood_obs_act = np.concatenate([ood_obs, ood_action], axis=1)




    def __getitem__(self, index):
        ood_obs_act = self.ood_obs_act[index % len(self.ood_obs_act)]
        id_obs_act = self.id_obs_act[index % len(self.ood_obs_act)]

        return {'ood_obs_act' : ood_obs_act, 'id_obs_act' : id_obs_act}

    def __len__(self):
        return len(self.ood_obs_act)
