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
parser.add_argument("--ood_test", type=bool, default=True, help="designate task name")
opts = parser.parse_args()

if opts.ood_test == False:
    path = './uncertainty_modeling/rl_uncertainty/mc_dropout/model'
else:
    path = './uncertainty_modeling/rl_uncertainty/mc_dropout/ood_model'

os.makedirs('{}/{}'.format(path, opts.env_name), exist_ok = True)
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
# Tensor = torch.Tensor

kwargs = {"dimensions": [200, 50, 50, 50],
              "output_dim": 1,
              "input_dim": 1}
args = list()
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

    ## Choose the training model
    model = FlattenMlp_Dropout(
        input_size=input_size,
        output_size=1,
        hidden_sizes=[256, 256],
    ).cuda()

    model.load_state_dict(torch.load("{}/{}/model_100.pt".format(path, opts.env_name)))  # if not handling ensemble

    for i, data in enumerate(dataloader):
        id_obs_act = Variable(data['id_obs_act'].type(Tensor))
        ood_obs_act = Variable(data['ood_obs_act'].type(Tensor))
        # if i == 0 :
        with torch.no_grad():
            ## Load testing dataset
            id_trajectories, ood_trajectories = [], []
            ## Iterative test for each model
            for i in range(10):
                id_output_ = model(id_obs_act).cpu().numpy().T
                ood_output_ = model(ood_obs_act).cpu().numpy().T
                id_trajectories.append(id_output_[:1, :])
                ood_trajectories.append(ood_output_[:1, :])
            id_trajectories = np.vstack(id_trajectories)
            ood_trajectories = np.vstack(ood_trajectories)

            # id_sigma = np.std(id_trajectories, axis=0)
            # ood_sigma = np.std(ood_trajectories, axis=0)
            id_sigma = np.mean(id_trajectories**2, axis=0) - np.mean(id_trajectories, axis=0) ** 2
            ood_sigma = np.mean(ood_trajectories ** 2, axis=0) - np.mean(ood_trajectories, axis=0) ** 2

            print('id_sigma : {}, ood_sigma : {}'.format(np.mean(id_sigma), np.mean(ood_sigma)))

test()