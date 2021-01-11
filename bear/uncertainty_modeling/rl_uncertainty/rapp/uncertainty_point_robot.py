from uncertainty_modeling.rl_uncertainty.model import *
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
from rlkit.torch.networks import FlattenMlp_Dropout
import numpy as np
import argparse
from rlkit.envs.read_hdf5 import get_dataset, qlearning_dataset
import gym
import d4rl
import numpy as np
from torch.utils.data import Dataset


class Point_Dataset(Dataset):
    def __init__(self):
        file_path = '/home/user/Documents/Workspace-Changyeop/Workspace/AdvancedDL/AI602_Project/bear/offline_buffer_itr_140.hdf5'  # offline_buffer_itr_140
        # Read the saved replay buffer
        data_dict = get_dataset(file_path)
        self.feature_sub = np.hstack([data_dict['observations'], data_dict['actions']])  # [::10]

    def __getitem__(self, index):
        obs_act = self.feature_sub[index % len(self.feature_sub)]

        return {'obs_act' : obs_act}

    def __len__(self):
        return len(self.feature_sub)


parser = argparse.ArgumentParser()
parser.add_argument("--env_name", type=str, default='point_robot', help="designate task name")
parser.add_argument("--ood_test", type=bool, default=False, help="designate task name")
opts = parser.parse_args()

path = './uncertainty_modeling/rl_uncertainty/rapp/model'

os.makedirs('{}/{}'.format(path, opts.env_name), exist_ok = True)
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

def train():
    epoch = 2000 # default : 3000
    qf_criterion = torch.nn.MSELoss()
    dataloader = DataLoader(
        Point_Dataset(),
        batch_size=400,
        shuffle=True,
        num_workers= 8,
    )

    ## Choose the training model
    model = RaPP(4).cuda()

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