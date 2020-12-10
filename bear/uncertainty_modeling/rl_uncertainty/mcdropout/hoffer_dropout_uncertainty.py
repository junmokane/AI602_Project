from model import *
import torch
from torch.utils.data import DataLoader
from hoffer_datasets import ScatterDataset, GymDataset
from torch.autograd import Variable
import os
from bear.rlkit.torch.networks import FlattenMlp_Dropout
import numpy as np

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.makedirs('dropout_half_exp_256', exist_ok = True)
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
        model = FlattenMlp_Dropout(
            input_size=14,
            output_size=1,
            hidden_sizes=[256, 256],
        )

        print(model)

        ## Choose the optimizer to train
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
                torch.save(model.state_dict(), './dropout_half_exp_256/rl_dropout_%d.pt' % (ep))
