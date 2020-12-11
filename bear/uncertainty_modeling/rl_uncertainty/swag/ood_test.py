from uncertainty_modeling.rl_uncertainty.model import *
import torch
from torch.utils.data import DataLoader
from uncertainty_modeling.rl_uncertainty.test_datasets import GymDataset
from torch.autograd import Variable
import os
from rlkit.torch.networks import FlattenMlp_Dropout
import numpy as np
import argparse
import gym
import d4rl

parser = argparse.ArgumentParser()
parser.add_argument("--env_name", type=str, default='halfcheetah-expert-v0', help="designate task name")
parser.add_argument("--ood_test", type=bool, default=False, help="designate task name")
opts = parser.parse_args()

if opts.ood_test == False:
    path = './uncertainty_modeling/rl_uncertainty/swag/model'
else:
    path = './uncertainty_modeling/rl_uncertainty/swag/ood_model'

os.makedirs('{}/{}'.format(path, opts.env_name), exist_ok = True)
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
# Tensor = torch.Tensor

discount = 0.99


def test():
    env = gym.make(opts.env_name)
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size
    input_size = obs_dim + action_dim
    ## Choose the trained model
    dataloader = DataLoader(
        # ScatterDataset(path='reg_data/test_data.npy'),
        GymDataset(env, opts.ood_test, opts.env_name),
        batch_size=400,
        shuffle=True,
        num_workers=8,
    )
    kwargs = {"dimensions": [200, 50, 50, 50],
              "output_dim": 1,
              "input_dim": input_size}
    args = list()

    ## Choose the training model
    model = SWAG(RegNetBase, subspace_type="pca", *args, **kwargs,
                      subspace_kwargs={"max_rank": 10, "pca_rank": 10})
    model.cuda()

    model.load_state_dict(torch.load("{}/{}/swag_model_100.pt".format(path, opts.env_name)))  # if not handling ensemble

    id_temp = []
    od_temp = []
    count = 0
    for i, data in enumerate(dataloader):
        id_obs_act = Variable(data['id_obs_act'].cuda())
        ood_obs_act = Variable(data['ood_obs_act'].cuda())
        # if i == 0 :
        count = count+1
        with torch.no_grad():
            ## Load testing dataset
            id_trajectories, ood_trajectories = [], []
            ## Iterative test for each model
            for i in range(10):
                model.sample(scale=10.)
                id_output_ = model(id_obs_act).cpu().numpy().T
                ood_output_ = model(ood_obs_act).cpu().numpy().T
                id_trajectories.append(id_output_[:1, :])
                ood_trajectories.append(ood_output_[:1, :])
            id_trajectories = np.vstack(id_trajectories)
            ood_trajectories = np.vstack(ood_trajectories)

            # id_sigma = np.std(id_trajectories, axis=0)
            # ood_sigma = np.std(ood_trajectories, axis=0)
            print(id_trajectories.shape)
            print(ood_trajectories.shape)
            exit()
            id_sigma = np.mean(id_trajectories**2, axis=0) - np.mean(id_trajectories, axis=0) ** 2
            ood_sigma = np.mean(ood_trajectories ** 2, axis=0) - np.mean(ood_trajectories, axis=0) ** 2
            id_temp.append(id_sigma)
            od_temp.append(ood_sigma)
            print('id_sigma : {}, ood_sigma : {}'.format(np.sum(id_temp)/count, np.sum(od_temp)/count))
        print('mean_id_sigma : {}, mean_ood_sigma : {}'.format(np.mean(id_temp), np.mean(od_temp)))

test()