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
from uncertainty_modeling.rl_uncertainty.rank1.r1bnn import Model

parser = argparse.ArgumentParser()
parser.add_argument("--env_name", type=str, default='halfcheetah-expert-v0', help="designate task name")
parser.add_argument("--ood_test", type=bool, default=False, help="designate task name")
opts = parser.parse_args()
if opts.ood_test == False:
    path = './uncertainty_modeling/rl_uncertainty/rank1/model'
else:
    path = './uncertainty_modeling/rl_uncertainty/rank1/ood_model'
os.makedirs('{}/{}'.format(path, opts.env_name), exist_ok = True)
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
# Tensor = torch.Tensor
discount = 0.99

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
    model = Model(x_dim=input_size, h_dim=10, y_dim=1, n=10).cuda()

    print(model)

    ## Choose the optimizer to train
    # optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.95, weight_decay=0.) # default
    # optim = torch.optim.Adam(model.parameters(), lr=1e-2)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_buffer = []

    for ep in range(epoch):
        for i, data in enumerate(dataloader):
            obs_act = Variable(data['obs_act'].type(Tensor))
            next_obs_act = Variable(data['next_obs_act'].type(Tensor))
            rewards = Variable(data['rewards'].type(Tensor))
            terminals = Variable(data['terminals'].type(Tensor))

            # loss, output, stats = criterion(model, input_, target_) # default

            target_q_values = model(next_obs_act).detach()
            rewards = rewards.repeat(10, 1, 1).squeeze(2)
            terminals = terminals.repeat(10, 1, 1).squeeze(2)
            y_target = rewards + (1. - terminals) * discount * target_q_values
            y_target = y_target.detach()
            y_pred = model(obs_act)
            loss = qf_criterion(y_pred, y_target)

            optim.zero_grad()
            loss.backward()
            optim.step()

            loss_buffer.append(loss.item())
        print('[Epoch : %d/%d] [loss : %f] ' % (ep, epoch, np.mean(np.array(loss_buffer))))

        if ep % 20 == 0:
            torch.save(model.state_dict(), '{}/{}/model_{}.pt'.format(path, opts.env_name, ep))

    test()


def test():
    ## Choose the trained model
    # model = RegNetBase(*args, **kwargs).type(Tensor) # default
    # model = FCN().type(Tensor)
    model = MC_Dropout_Model(input_dim=1, output_dim=1, num_units=200, drop_prob=0.5).type(Tensor)
    # model = SWAG(RegNetBase, subspace_type="pca", *args, **kwargs,
    #              subspace_kwargs={"max_rank": 10, "pca_rank": 10}).type(Tensor)

    with torch.no_grad():
        ## Load testing dataset
        data = np.load('reg_data/test_data.npy')
        z = np.reshape(np.linspace(-3, 3, 100), [-1, 1])
        input_ = torch.from_numpy(z.astype(np.float32)).type(Tensor)

        trajectories = []
        ## Iterative test for each model
        for i in range(10):
            # model.load_state_dict(torch.load("./save/ensemble_" + str(i) + ".pt")) # default
            model.load_state_dict(torch.load("dropout_" + str(0) + ".pt")) # if not handling ensemble
            # model.load_state_dict(torch.load("test_ensemble_" + str(i) + ".pt")) # if not handling ensemble
            # model.load_state_dict(torch.load("ckpts/swag_checkpoint0-300.pt")["state_dict"]) # if not handling ensemble
            # print(model.subspace.cov_mat_sqrt)
            # model.sample(scale=10.)
            output_ = model(input_).cpu().numpy().T
            # trajectories.append(output_) # default
            trajectories.append(output_[:1, :])
        trajectories = np.vstack(trajectories)
        plot_predictive(data, trajectories, z, title="Swag_Confidence 95%")

## What to do for running
train()
# test()