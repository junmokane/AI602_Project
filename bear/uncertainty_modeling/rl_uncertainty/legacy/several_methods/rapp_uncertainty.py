from model import *
import torch
from torch.utils.data import DataLoader
from datasets import ScatterDataset, GymDataset
from torch.autograd import Variable
import os
from bear.rlkit.torch.networks import FlattenMlp
import numpy as np

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"

# Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
Tensor = torch.Tensor

kwargs = {"dimensions": [200, 50, 50, 50],
              "output_dim": 1,
              "input_dim": 1}
args = list()
Num_ensemble = 1
discount = 0.99

def train():
    epoch = 3000 # default : 3000
    qf_criterion = torch.nn.MSELoss()
    # criterion = torch.nn.MSELoss()
    dataloader = DataLoader(
        # ScatterDataset(path='reg_data/test_data.npy'),
        GymDataset(),
        # batch_size=50, # default
        batch_size=400,
        shuffle=True,
        num_workers= 8,
    )

    for md in range(Num_ensemble):
        print('Training Model Num : %d'%(md))
        model = RaPP(23)
        print(model)

        loss_buffer = []

        ## Choose the optimizer to train
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)
        for ep in range(epoch):
            for i, data in enumerate(dataloader):
                obs_act = Variable(data['obs_act'].type(Tensor))
                # next_obs_act = Variable(data['next_obs_act'].type(Tensor))
                # rewards = Variable(data['rewards'].type(Tensor))
                # terminals = Variable(data['terminals'].type(Tensor))

                y_pred = model(obs_act)
                loss = qf_criterion(y_pred, obs_act)

                optim.zero_grad()
                loss.backward()
                optim.step()

                y_repr = torch.mean(y_pred)
                # print('[Epoch : %d/%d] [Batch : %d/%d] [loss : %f] [q : %f]' % (ep, epoch, i, len(dataloader), loss.item(), y_repr.item()))
                loss_buffer.append(loss.item())
            print('[Epoch : %d/%d] [loss : %f]' % (ep, epoch, np.mean(np.array(loss_buffer))))
        torch.save(model.state_dict(), 'rl_rapp_%d.pt'%(md))


## What to do for running
train()
