from model import *
import torch
from torch.utils.data import DataLoader
from walker_datasets import ScatterDataset, GymDataset
from torch.autograd import Variable
import os
from bear.rlkit.torch.networks import FlattenMlp_Dropout
import numpy as np

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"
os.makedirs('dropout_walker_exp_256', exist_ok = True)
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
        ## Choose the training model
        # model = FlattenMlp_Dropout(
        #     input_size=23,
        #     output_size=1,
        #     hidden_sizes=[256, 256],
        # )
        model = MC_Dropout_Model(input_dim=23, output_dim=1, num_units=256, drop_prob=0.5).type(Tensor)

        print(model)

        ## Choose the optimizer to train
        optim = torch.optim.Adam(model.parameters(), lr=1e-5)
        loss_buffer = []

        for ep in range(epoch):
            for i, data in enumerate(dataloader):
                obs_act = Variable(data['obs_act'].type(Tensor))
                next_obs_act = Variable(data['next_obs_act'].type(Tensor))
                rewards = Variable(data['rewards'].type(Tensor))
                terminals = Variable(data['terminals'].type(Tensor))

                # loss, output, stats = criterion(model, input_, target_) # default

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
                torch.save(model.state_dict(), './dropout_walker_exp_256/rl_dropout_%d.pt' % (ep))


## What to do for running
train()
