import argparse
import numpy as np
import torch
import torch.nn.init as init
import torch.nn as nn
import torch.distributions as D
import torch.optim as optim
import matplotlib.pyplot as plt

from uncertainty_modeling.mdn.mixture_same_family import MixtureSameFamily
from uncertainty_modeling.gen_data import gen_datagrid, plot_meshgrid
from uncertainty_modeling.mdn.model import MDN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu'


def check_cuda():
    print('The CUDA version of pyTorch is ' + str(torch.version.cuda))
    print(str(torch.cuda.device_count()) + ' GPUs are available')
    if not torch.cuda.is_available():
        print("pyTorch is not using GPU")
        exit()


def init_weights(m):
    if type(m) == nn.Linear:
        init.normal_(m.weight, std=0.01)
        m.bias.data.fill_(0.0)


def GMM(pi, mu, sigma):
    mix = D.Categorical(pi)
    comp = D.Independent(D.Normal(mu, sigma), 1)
    gmm = MixtureSameFamily(mix, comp)
    return gmm


def calc_uncertainty(meshgrid_data_lin, model):
    model.eval()
    pi, mu, sigma = model(meshgrid_data_lin)  # (b,k), (b,k,y_dim), (b,k,y_dim)
    b, k, y_dim = mu.size()

    # aleatoric uncertainty
    pi_axis = pi[..., None]
    al_un = torch.sum(pi_axis * sigma, dim=1)  # (b, y_dim)

    # epistemic uncertainty
    mu_average = torch.sum(pi_axis * mu, dim=1)  # (b,y_dim)
    mu_average = mu_average[:, None, :]  # (b,1,y_dim)
    mu_average = mu_average.repeat_interleave(k, dim=1)  # (b,k,y_dim)
    mu_diff_sq = (mu - mu_average) ** 2
    ep_un = torch.sum(pi_axis * mu_diff_sq, dim=1)  # (b, y_dim)

    return al_un, ep_un


if __name__ == "__main__":
    check_cuda()
    parser = argparse.ArgumentParser(description='MDN Specification')
    parser.add_argument('--lr', default=1e-4, type=float)
    args = parser.parse_args()

    # generate data
    start, end, step = -1, 1, 101
    train_data, index, meshgrid_data_lin = gen_datagrid(start=start, end=end, step=step,
                                                        plot=False,
                                                        xrange=list(range(30, 70, 2)),
                                                        size='big',
                                                        type='random')
    train_data = train_data.to(device)
    labels = 2 * train_data[:, 0:1] - train_data[:, 1:2].to(device)
    meshgrid_data_lin = meshgrid_data_lin.type(torch.float32).to(device)
    print("training data: ", train_data.dtype, train_data.shape, len(train_data))
    print("labels: ", labels.dtype, labels.shape, len(labels))
    print("all data: ", meshgrid_data_lin.dtype, meshgrid_data_lin.shape, len(meshgrid_data_lin))

    # define model
    model = MDN(x_dim=2, y_dim=1, k=1, h_dims=[32, 32])
    model = model.to(device)
    model.apply(init_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print(model)

    # training
    for i in range(20000):
        pi, mu, sigma = model(train_data)  # (b,k), (b,k,y_dim), (b,k,y_dim)
        gmm = GMM(pi, mu, sigma)
        logp = gmm.log_prob(labels)
        nll = -torch.mean(logp)

        optimizer.zero_grad()
        nll.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(i + 1, nll.item())


    '''
    Calculate and plot uncertainty. 
    Notice that this part is only valid when y_dim = 1.
    '''
    with torch.no_grad():
        al_un, ep_un = calc_uncertainty(meshgrid_data_lin, model)
        al_un = np.reshape(al_un.cpu().numpy(), (step, step))
        ep_un = np.reshape(ep_un.cpu().numpy(), (step, step))
        un = al_un + ep_un
        print(np.max(un), np.min(un))

        for i in range(len(train_data)):
            pi, mu, sigma = model(train_data[i:i+1])
            ind = torch.argmax(pi)
            print('input', train_data[i], 'pi', pi[0, ind], 'mu', mu[0, ind],
                  'sigma', sigma[0, ind], 'label', labels[i])

        '''
        al_un, ep_un = calc_uncertainty(train_data, model)
        al_un = al_un.cpu().numpy()
        ep_un = ep_un.cpu().numpy()
        un = al_un + ep_un
        print(np.max(un), np.min(un))

        pi, mu, sigma = model(torch.from_numpy(np.array([[-1.0, 1.0]])).type(torch.float32))
        ind = torch.argmax(pi)
        print('pi', pi[0, ind], 'mu', mu[0, ind], 'sigma', sigma[0, ind])
        '''

        fig, (a2) = plt.subplots(1, 1)
        im2 = a2.imshow(un, cmap="gray", extent=[start, end, start, end], origin="lower")
        a2.set_title("uncertainty plot")
        plot_meshgrid(a2, meshgrid_data_lin, index)

        plt.show()

