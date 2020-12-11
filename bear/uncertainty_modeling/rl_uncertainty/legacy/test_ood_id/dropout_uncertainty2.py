from .model import *
import torch
from torch.utils.data import DataLoader
from .ood_datasets import ScatterDataset, GymDataset, GymDataset_test
from torch.autograd import Variable
import os
from rlkit.torch.networks import FlattenMlp_Dropout
import numpy as np

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.makedirs('dropout_128', exist_ok = True)
# Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
Tensor = torch.Tensor



kwargs = {"dimensions": [200, 50, 50, 50],
              "output_dim": 1,
              "input_dim": 1}
args = list()
Num_ensemble = 1
discount = 0.99

def train():
    epoch = 2000 # default : 3000
    qf_criterion = torch.nn.MSELoss()
    dataloader = DataLoader(
        GymDataset(),
        batch_size=400,
        shuffle=True,
        num_workers= 8,
    )

    for md in range(Num_ensemble):
        print('Training Model Num : %d'%(md))

        model = FlattenMlp_Dropout(
            input_size=23,
            output_size=1,
            hidden_sizes=[256, 256],
        )

        ## Choose the optimizer to train

        optim = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_buffer = []

        for ep in range(epoch):
            for i, data in enumerate(dataloader):
                obs_act = Variable(data['obs_act'].type(Tensor))
                next_obs_act = Variable(data['next_obs_act'].type(Tensor))
                rewards = Variable(data['rewards'].type(Tensor))
                terminals = Variable(data['terminals'].type(Tensor))

                target_q_values = model(next_obs_act).detach()
                y_target = rewards + (1. - terminals) * discount * target_q_values
                y_target = y_target.detach()
                y_pred = model(obs_act)
                loss = qf_criterion(y_pred, y_target)

                optim.zero_grad()
                loss.backward()
                optim.step()

                # print('[Epoch : %d/%d] [Batch : %d/%d] [loss : %f] [q : %f]' % (ep, epoch, i, len(dataloader), loss.item(), y_repr.item()))
                loss_buffer.append(loss.item())
            print('[Epoch : %d/%d] [loss : %f] ' % (ep, epoch, np.mean(np.array(loss_buffer))))
            if ep % 20 == 0:
                torch.save(model.state_dict(), './dropout_128/rl_dropout_%d.pt' % (ep))

    test()


def test():
    ## Choose the trained model
    model = FlattenMlp_Dropout(
            input_size=23,
            output_size=1,
            hidden_sizes=[128, 128],
        )

    dataloader = DataLoader(
        GymDataset_test(),
        batch_size=1000,
        shuffle=True,
        num_workers=8,
    )

    model.load_state_dict(torch.load("./dropout_128/rl_dropout_" + str(60) + ".pt"))  # if not handling ensemble

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

## What to do for running
train()
# test()