import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F



class FCN(nn.Module):
    def __init__(self, input_size = 1, fc1_units = 200, fc2_units=200, output_size = 1):
        super(FCN, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(input_size, fc1_units), nn.Dropout(0.5), nn.ReLU(True))
        self.fc2 = nn.Sequential(nn.Linear(fc1_units, fc2_units), nn.Dropout(0.5), nn.ReLU(True))
        self.fc3 = nn.Linear(fc2_units, output_size)


    def forward(self, input_):
        x = self.fc1(input_)
        x = self.fc2(x)
        # x = self.fc3(x * weight)
        x = self.fc3(x)

        return x


class MC_Dropout_Model(nn.Module):
    def __init__(self, input_dim, output_dim, num_units, drop_prob):
        super(MC_Dropout_Model, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.drop_prob = drop_prob

        # network with two hidden and one output layer
        self.layer1 = nn.Linear(input_dim, num_units)
        self.layer2 = nn.Linear(num_units, 2 * output_dim)

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.view(-1, self.input_dim)

        x = self.layer1(x)
        x = self.activation(x)

        x = F.dropout(x, p=self.drop_prob, training=True)

        x = self.layer2(x)

        return x


class MDropout(torch.nn.Module):
    def __init__(self, dim, p, inplace=False):
        super(MDropout, self).__init__()
        self.dim = dim
        self.p = p
        self.inplace = inplace
        self.register_buffer('mask', torch.ones(dim, dtype=torch.long))

    def forward(self, input):
        if self.training:
            return torch.nn.functional.dropout(input, self.p, self.training, self.inplace)
        else:
            return input * self.mask.float().view(1, -1) * 1.0 / (1.0 - self.p)

    def sample(self):
        self.mask.bernoulli_(1.0 - self.p)


def sample_masks(module):
    if isinstance(module, MDropout):
        module.sample()


class SplitDim(nn.Module):
    def __init__(self, nonlin_col=1, nonlin_type=torch.nn.functional.softplus, correction=True):
        super(SplitDim, self).__init__()
        self.nonlinearity = nonlin_type
        self.col = nonlin_col

        if correction:
            self.var = torch.nn.Parameter(torch.zeros(1))
        else:
            # equivalent to about 3e-7 when using softplus
            self.register_buffer('var', torch.ones(1, requires_grad=False) * -15.)

        self.correction = correction

    def forward(self, input):
        transformed_output = self.nonlinearity(input[:, self.col])

        transformed_output = (transformed_output + self.nonlinearity(self.var))
        stack_list = [input[:, :self.col], transformed_output.view(-1, 1)]
        if self.col + 1 < input.size(1):
            stack_list.append(input[:, (self.col + 1):])

        # print(self.nonlinearity(self.var).item(), transformed_output.mean().item())
        output = torch.cat(stack_list, 1)
        return output

class RegNetBase(nn.Sequential):
    def __init__(self, dimensions, input_dim=1, output_dim=1, dropout=None, apply_var=True):
        super(RegNetBase, self).__init__()
        self.dimensions = [input_dim, *dimensions, output_dim]
        for i in range(len(self.dimensions) - 1):
            if dropout is not None and i > 0:
                self.add_module('dropout%d' % i, MDropout(self.dimensions[i], p=dropout))
            self.add_module('linear%d' % i, torch.nn.Linear(self.dimensions[i], self.dimensions[i + 1]))
            if i < len(self.dimensions) - 2:
                self.add_module('relu%d' % i, torch.nn.ReLU())

        if output_dim == 2:
            self.add_module('var_split', SplitDim(correction=apply_var))

    def forward(self, x, output_features=False):
        if not output_features:
            return super().forward(x)
        else:
            print(self._modules.values())
            print(list(self._modules.values())[:-2])
            for module in list(self._modules.values())[:-3]:
                x = module(x)
                print(x.size())
            return x

def log_gaussian_loss(output, target, sigma, no_dim):
    exponent = -0.5 * (target - output) ** 2 / sigma ** 2
    log_coeff = -no_dim * torch.log(sigma) - 0.5 * no_dim * np.log(2 * np.pi)

    return - (log_coeff + exponent).sum()


class GaussianLikelihood:
    def __init__(self, noise_var=0.5):
        self.noise_var = noise_var
        self.mse = torch.nn.functional.mse_loss

    def __call__(self, model, input, target, weight=None):

        output = model(input)
        mse = self.mse(output, target)
        loss = mse / (2 * self.noise_var)

        return loss, output, {"mse": mse}
