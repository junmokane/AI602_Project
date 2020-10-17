import torch
import torch.nn as nn
import torch.nn.functional as F


class MDN(nn.Module):
    def __init__(self, x_dim, y_dim, k, h_dims):
        super(MDN, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.k = k  # # of mixtures
        self.h_dims = h_dims  # hidden layers
        self.feature = h_dims[-1]  # extracted feature

        # feature extraction part of NN
        self.linears = nn.ModuleList()
        dim_list = [self.x_dim] + self.h_dims
        for i in range(len(dim_list) - 1):
            self.linears.append(nn.Linear(dim_list[i], dim_list[i+1]))

        # pi, mu, sigma
        self.pi = nn.Sequential(nn.Linear(self.feature, self.k), nn.Softmax(dim=1))
        self.mu = nn.Linear(self.feature, self.y_dim * self.k)
        self.sigma = nn.Linear(self.feature, self.y_dim * self.k)

    def forward(self, x):
        for linear in self.linears:
            x = linear(x)
            x = F.tanh(x)
        pi = self.pi(x)
        mu = self.mu(x)
        mu = mu.view(-1, self.k, self.y_dim)
        sigma = torch.exp(self.sigma(x))
        sigma = sigma.view(-1, self.k, self.y_dim)
        return pi, mu, sigma


if __name__ == "__main__":
    input_sample = torch.zeros(100, 2).cuda()
    model = MDN(x_dim=2, y_dim=1, k=5, h_dims=[32, 32]).cuda()
    print("str : ", model)
    pi, mu, sigma = model(input_sample)
    print(pi.size(), mu.size(), sigma.size())
