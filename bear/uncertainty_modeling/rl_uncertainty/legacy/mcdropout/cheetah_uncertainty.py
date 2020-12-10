from model import *
import torch
from torch.utils.data import DataLoader
from cheetah_datasets import ScatterDataset, GymDataset
from torch.autograd import Variable
import os
from bear.rlkit.torch.networks import FlattenMlp_Dropout
import numpy as np

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
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
    criterion = GaussianLikelihood(noise_var=1.) # = MSELoss / (2 * noise_var) # default
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
        ## Choose the training model
        model = FlattenMlp_Dropout(
            input_size=23,
            output_size=1,
            hidden_sizes=[256, 256],
        )

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
        # torch.save(model.state_dict(), 'test_ensemble_%d.pt'%(md))
            if ep % 20 == 0:
                torch.save(model.state_dict(), './dropout_half_exp_256/rl_dropout_%d.pt' % (ep))
        # torch.save(model.state_dict(), 'swag_%d.pt'%(md))

        # utils.save_checkpoint(
        #              dir="ckpts",
        #              epoch=epoch,
        #              name="swag_checkpoint"+str(0),
        #              state_dict=swag_model.state_dict(),
        #         )

        # torch.save(model.state_dict(), 'dropout_%d.pt'%(md))

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