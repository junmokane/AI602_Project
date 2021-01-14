# Perform the necessary imports
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
# from rlkit.envs.read_hdf5 import get_dataset, qlearning_dataset
import matplotlib.pyplot as plt
import numpy as np
import h5py
import torch
from uncertainty_modeling.rl_uncertainty.model import *
import gym
import d4rl

def get_diffs(x, model, batch_size=256):
    model.eval()
    with torch.no_grad():
        batchified = x.split(batch_size)
        stacked = []
        for _x in batchified:
            model.eval()
            diffs = []
            _x = _x.to(next(model.parameters()).device).float()
            x_tilde = model(_x)
            diffs.append((x_tilde - _x).cpu())

            for layer in model.enc_layer_list:
                _x = layer(_x)
                x_tilde = layer(x_tilde)
                diffs.append((x_tilde - _x).cpu())

            stacked.append(diffs)

        stacked = list(zip(*stacked))
        diffs = [torch.cat(s, dim=0).numpy() for s in stacked]

    return diffs

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
file_path2 = '/home/user/Documents/Workspace-Changyeop/Workspace/AdvancedDL/AI602_Project/bear/offline_buffer_itr_140.hdf5' # offline_buffer_itr_140
# Read the saved replay buffer
data_dict = get_dataset(file_path)
data_dict2 = get_dataset(file_path2)

print(data_dict['observations'].shape)
print(data_dict['actions'].shape)
# feature_sub = np.hstack([data_dict['observations'][::10], data_dict['actions'][::10]]) #
# feature_sub2 = np.hstack([data_dict2['observations'][::10], data_dict2['actions'][::10]])
feature_sub = np.hstack([data_dict['observations'], data_dict['actions']]) #
feature_sub2 = np.hstack([data_dict2['observations'], data_dict2['actions']])
print(np.max(data_dict['observations']))
print(np.min(data_dict['observations']))
print(np.max(data_dict['actions']))
print(np.min(data_dict['actions']))

a = np.linspace(0, 1, 7)
b = np.linspace(0, 1, 7)
c = np.linspace(0, 1, 7)
d = np.linspace(0, 1, 7)
av, bv, cv, dv = np.meshgrid(a,b,c,d)

meshgrid_data = torch.from_numpy(np.stack([av,bv,cv,dv], axis=-1))
meshgrid_data = np.reshape(meshgrid_data, [-1, 4])


model = RaPP(4).cuda()

model.load_state_dict(torch.load("/home/user/Documents/Workspace-Changyeop/Workspace/AdvancedDL/AI602_Project/bear/uncertainty_modeling/rl_uncertainty/rapp/model/point-robot/model_1980.pt"))  # if not handling ensemble

# id_dif = get_diffs(meshgrid_data, model)
id_dif = get_diffs(torch.from_numpy(feature_sub2), model)


id_difs = torch.cat([torch.from_numpy(i) for i in id_dif], dim=-1).numpy()
id_dif = (id_difs**2).mean(axis=1)

print(np.mean(id_dif), np.max(id_dif), np.min(id_dif))

#
#
# # feature = np.vstack([feature_sub, feature_sub2, meshgrid_data])
# feature = np.vstack([expert_data, medium_data])
#
# model = TSNE(learning_rate=10)
# transformed = model.fit_transform(feature)
#
#
# xs = transformed[:,0]
# ys = transformed[:,1]
#
# # plt.scatter(xs[1600:3200],ys[1600:3200],color="g")
# plt.scatter(xs[:2000],ys[:2000],color="g")
# import seaborn as sns
# cmap = sns.diverging_palette(240, 10, l=65, center="dark", as_cmap=True)
# vmin= np.min(id_dif)
# vmax= np.max(id_dif)
# sc = plt.scatter(xs[2000:],ys[2000:], c=id_dif,vmin=vmin, vmax=vmax, cmap=cmap)
# plt.colorbar(sc)
# plt.show()
