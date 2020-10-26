from model import *
import torch
from torch.utils.data import DataLoader
from datasets import ScatterDataset
from torch.autograd import Variable
from visualization import plot_predictive
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

kwargs = {"dimensions": [200, 50, 50, 50],
              "output_dim": 1,
              "input_dim": 1}
args = list()
Num_ensemble = 10

def train():
    epoch = 100 # default : 3000
    criterion = GaussianLikelihood(noise_var=1.) # = MSELoss / (2 * noise_var)
    # criterion = torch.nn.MSELoss()
    dataloader = DataLoader(
        ScatterDataset(path='reg_data/data2.npy'),
        batch_size=50,
        shuffle=True,
        num_workers= 8,
    )

    for md in range(Num_ensemble):
        print('Training Model Num : %d'%(md))
        ## Choose the training model
        model = RegNetBase(*args, **kwargs).type(Tensor) # Simple 5-layer fully-connected network
        print(model)
        # model = FCN().type(Tensor)
        # model = MC_Dropout_Model(input_dim=1, output_dim=1, num_units=200, drop_prob=0.5).type(Tensor)

        ## Choose the optimizer to train
        optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.95, weight_decay=0.)
        # optim = torch.optim.Adam(model.parameters(), lr=1e-2)
        # optim = torch.optim.SGD(model.parameters(), lr=1e-4, weight_decay=1e-2)

        for ep in range(epoch):
            for i, data in enumerate(dataloader):
                input_ = Variable(data['input'].type(Tensor))
                target_ = Variable(data['label'].type(Tensor))

                input_ = torch.reshape(input_, [-1, 1])
                target_ = torch.reshape(target_, [-1, 1])

                loss, output, stats = criterion(model, input_, target_, weight_)

                optim.zero_grad()
                loss.backward()
                optim.step()
                print('[Epoch : %d/%d] [Batch : %d/%d] [loss : %f]' % (ep, epoch, i, len(dataloader), loss.item()))
        torch.save(model.state_dict(), 'ensemble_%d.pt'%(md))

    test()


def test():
    ## Choose the trained model
    model = RegNetBase(*args, **kwargs).type(Tensor)
    # model = FCN().type(Tensor)
    # model = MC_Dropout_Model(input_dim=1, output_dim=1, num_units=200, drop_prob=0.5).type(Tensor)

    with torch.no_grad():
        ## Load testing dataset
        data = np.load('reg_data/data.npy')
        z = np.reshape(np.linspace(-10, 10, 100), [-1, 1])
        input_ = torch.from_numpy(z.astype(np.float32)).type(Tensor)

        trajectories = []
        ## Iterative test for each model
        for i in range(Num_ensemble):
            model.load_state_dict(torch.load("./save/ensemble_" + str(i) + ".pt"))
            # model.load_state_dict(torch.load("dropout_" + str(0) + ".pt")) # if not handling ensemble

            output_ = model(input_).cpu().numpy().T
            trajectories.append(output_)
        trajectories = np.vstack(trajectories)
        plot_predictive(data, trajectories, z, title="Confidence 95%")

## What to do for running
# train()
test()