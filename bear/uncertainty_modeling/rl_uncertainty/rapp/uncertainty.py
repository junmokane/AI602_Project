from uncertainty_modeling.rl_uncertainty.model import *
import torch
from torch.utils.data import DataLoader
from uncertainty_modeling.rl_uncertainty.datasets import GymDataset
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

def train():
    env = gym.make(opts.env_name)
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size
    input_size = obs_dim + action_dim

    epoch = 2000 # default : 3000
    qf_criterion = torch.nn.MSELoss()
    dataloader = DataLoader(
        # ScatterDataset(path='reg_data/test_data.npy'),
        GymDataset(env, opts.ood_test, opts.env_name),
        batch_size=400,
        shuffle=True,
        num_workers= 8,
    )

    ## Choose the training model
    model = RaPP(input_size).cuda()

    print(model)

    ## Choose the optimizer to train
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_buffer = []

    for ep in range(epoch):
        for i, data in enumerate(dataloader):
            obs_act = Variable(data['obs_act'].type(Tensor))
            y_pred = model(obs_act)
            loss = qf_criterion(y_pred, obs_act)

            optim.zero_grad()
            loss.backward()
            optim.step()

            loss_buffer.append(loss.item())
        print('[Epoch : %d/%d] [loss : %f] ' % (ep, epoch, np.mean(np.array(loss_buffer))))

        if ep % 20 == 0:
            torch.save(model.state_dict(), '{}/{}/model_{}.pt'.format(path, opts.env_name, ep))

train()