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
    path = './uncertainty_modeling/rl_uncertainty/rapp/model'
else:
    path = './uncertainty_modeling/rl_uncertainty/rapp/ood_model'

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

    ## Choose the training model
    model = RaPP(23)

    # model.load_state_dict(torch.load("{}/{}/model_80.pt".format(path, opts.env_name)))  # if not handling ensemble

    id_temp = []
    od_temp = []
    count = 0
    for i, data in enumerate(dataloader):
        id_obs_act = Variable(data['id_obs_act'])
        ood_obs_act = Variable(data['ood_obs_act'])
        # if i == 0 :
        count = count+1
        with torch.no_grad():
            ## Load testing dataset
            id_trajectories, ood_trajectories = [], []
            ## Iterative test for each mode
            # id_output_ = model(id_obs_act)
            # ood_output_ = model(ood_obs_act)
            # id_sigma = get_diffs(id_output_, model)
            # ood_sigma = get_diffs(ood_output_, model)

            id_dif = get_diffs(id_obs_act, model)
            ood_dif = get_diffs(ood_obs_act, model)
            id_difs = torch.cat([torch.from_numpy(i) for i in id_dif], dim=-1).numpy()
            ood_difs = torch.cat([torch.from_numpy(i) for i in ood_dif], dim=-1).numpy()
            print(id_difs.shape, ood_difs.shape) # (400, 49), (400, 49)
            id_dif = (id_difs**2).mean(axis=1)
            ood_dif = (ood_difs**2).mean(axis=1)
            print(id_dif.shape, ood_dif.shape)  # (400,), (400,) id_dif will be differnce between in-dist

            # print(len(id_sigma))
            # print(len(ood_sigma))
            print('id_sigma : {}, ood_sigma : {}'.format(np.mean(id_sigma), np.mean(ood_sigma)))
        # print('mean_id_sigma : {}, mean_ood_sigma : {}'.format(np.mean(id_temp), np.mean(od_temp)))


def get_diffs(x, model, batch_size=23):
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

test()