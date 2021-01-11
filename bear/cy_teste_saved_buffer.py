# from rlkit.envs.read_hdf5 import get_dataset, qlearning_dataset
import matplotlib.pyplot as plt
import numpy as np
import h5py

def get_keys(h5file):
    keys = []
    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)
    h5file.visititems(visitor)
    return keys

def get_dataset(h5path):
    dataset_file = h5py.File(h5path, 'r')
    data_dict = {k: dataset_file[k][:] for k in get_keys(dataset_file)}
    dataset_file.close()
    return data_dict


def qlearning_dataset(dataset):
    """
    Returns datasets formatted for use by standard Q-learning algorithms,
    with observations, actions, next_observations, rewards, and a terminal
    flag.
    Args:
        dataset: dataset to pass in for processing.
    Returns:
        A dictionary containing keys:
            observations: An N x dim_obs array of observations.
            actions: An N x dim_action array of actions.
            next_observations: An N x dim_obs array of next observations.
            rewards: An N-dim float array of rewards.
            terminals: An N-dim boolean array of "done" or episode termination flags.
    """
    N = dataset['rewards'].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []

    for i in range(N-1):
        obs = dataset['observations'][i]
        new_obs = dataset['observations'][i+1]
        action = dataset['actions'][i]
        reward = dataset['rewards'][i]
        done_bool = bool(dataset['terminals'][i])
        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)

    return {
        'observations': np.array(obs_),
        'actions': np.array(action_),
        'next_observations': np.array(next_obs_),
        'rewards': np.array(reward_),
        'terminals': np.array(done_),
    }


file_path = '/home/user/Documents/Workspace-Changyeop/Workspace/AdvancedDL/AI602_Project/bear/online_buffer.hdf5' # offline_buffer_itr_140
# Read the saved replay buffer
data_dict = get_dataset(file_path)
# Run a few quick sanity checks
for key in ['observations', 'actions', 'rewards', 'terminals']:
    assert key in data_dict, 'Dataset is missing key %s' % key

print(data_dict['observations'].shape,
      data_dict['actions'].shape,
      data_dict['rewards'].shape,
      data_dict['terminals'].shape)
qlearning_data = qlearning_dataset(data_dict)
print(qlearning_data['observations'].shape,
      qlearning_data['actions'].shape,
      qlearning_data['next_observations'].shape,
      qlearning_data['rewards'].shape,
      qlearning_data['terminals'].shape)

# plot data
states = data_dict['observations']
plt.plot(states[:, 1], states[:, 0], 'ro')
plt.show()