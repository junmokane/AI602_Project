from model import *
import torch
from torch.utils.data import DataLoader
from datasets import ScatterDataset, GymDataset
from torch.autograd import Variable
import os
# from bear.rlkit.torch.networks import FlattenMlp
import numpy as np

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.makedirs('swag', exist_ok = True)

# Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
Tensor = torch.Tensor

kwargs = {"dimensions": [200, 50, 50, 50],
              "output_dim": 1,
              "input_dim": 23}
args = list()
Num_ensemble = 1
discount = 0.99

def train():
    epoch = 150 # default : 3000
    qf_criterion = torch.nn.MSELoss()
    dataloader = DataLoader(
        # ScatterDataset(path='reg_data/test_data.npy'),
        GymDataset(),
        batch_size=400,
        shuffle=True,
        num_workers= 8,
    )

    for md in range(Num_ensemble):
        print('Training Model Num : %d'%(md))
        ## Choose the training model
        model = RegNetBase(*args, **kwargs).type(Tensor) # Simple 5-layer fully-connected network

        # swag part
        swag_model = SWAG(RegNetBase, subspace_type="pca", *args, **kwargs,
                  subspace_kwargs={"max_rank": 10, "pca_rank": 10}).type(Tensor)
        swag = True
        swag_start = 100
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

                target_q_values = model(next_obs_act).detach()
                y_target = rewards + (1. - terminals) * discount * target_q_values
                y_target = y_target.detach()
                y_pred = model(obs_act)
                loss = qf_criterion(y_pred, y_target)

                optim.zero_grad()
                loss.backward()
                optim.step()

                if swag and ep > swag_start:
                    swag_model.collect_model(model)

                # print('[Epoch : %d/%d] [Batch : %d/%d] [loss : %f] [q : %f]' % (ep, epoch, i, len(dataloader), loss.item(), y_repr.item()))
                loss_buffer.append(loss.item())
            print('[Epoch : %d/%d] [loss : %f] ' % (ep, epoch, np.mean(np.array(loss_buffer))))
            if ep % 20 == 0:
                utils.save_checkpoint(
                             dir="swag",
                             epoch=epoch,
                             name="rl_swag_checkpoint"+str(ep),
                             state_dict=swag_model.state_dict(),
                        )


    # test()

def test():
    ## Choose the trained model
    model = SWAG(RegNetBase, subspace_type="pca", *args, **kwargs,
                 subspace_kwargs={"max_rank": 10, "pca_rank": 10}).type(Tensor)

    with torch.no_grad():
        ## Load testing dataset
        # data = np.load('reg_data/test_data.npy')
        z = np.reshape(np.linspace(-3, 3, 100), [-1, 1])
        input_ = torch.from_numpy(z.astype(np.float32)).type(Tensor)

        trajectories = []
        ## Iterative test for each model
        for i in range(10):
            # model.load_state_dict(torch.load("ckpts/swag_checkpoint0-300.pt")["state_dict"]) # if not handling ensemble
            print(model.subspace.cov_mat_sqrt)
            model.sample(scale=10.)
            output_ = model(input_).cpu().numpy().T
            trajectories.append(output_[:1, :])
        trajectories = np.vstack(trajectories)
        # plot_predictive(data, trajectories, z, title="Swag_Confidence 95%")

## What to do for running
# train()
test()