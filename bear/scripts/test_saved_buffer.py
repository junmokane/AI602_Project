from rlkit.envs.read_hdf5 import get_dataset, qlearning_dataset
import matplotlib.pyplot as plt
import numpy as np

file_path = './data/sac-point-robot/2021_01_04_22_25_16_exp_0000_s_0/offline_buffer_itr_140.hdf5'
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