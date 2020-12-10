import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from utils import flatten, set_weights
from subspaces import Subspace


Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

def log_gaussian_loss(output, target, sigma, no_dim):
    exponent = -0.5 * (target - output) ** 2 / sigma ** 2
    log_coeff = -no_dim * torch.log(sigma) - 0.5 * no_dim * np.log(2 * np.pi)

    return - (log_coeff + exponent).sum()

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
        # self.layer2 = nn.Linear(num_units, 2 * output_dim) # for regression
        self.layer2 = nn.Linear(num_units, num_units)
        self.layer3 = nn.Linear(num_units, output_dim)

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.view(-1, self.input_dim)

        x = self.layer1(x)
        x = self.activation(x)
        x = F.dropout(x, p=self.drop_prob, training=True)

        x = self.layer2(x)
        x = self.activation(x)
        # x = F.dropout_128(x, p=self.drop_prob, training=True)

        x = self.layer3(x)

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
                self.add_module('dropout_128%d' % i, MDropout(self.dimensions[i], p=dropout))
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


class SWAG(torch.nn.Module):

    def __init__(self, base, subspace_type,
                 subspace_kwargs=None, var_clamp=1e-6, *args, **kwargs):
        super(SWAG, self).__init__()

        self.base_model = base(*args, **kwargs)
        self.num_parameters = sum(param.numel() for param in self.base_model.parameters())

        self.register_buffer('mean', torch.zeros(self.num_parameters))
        self.register_buffer('sq_mean', torch.zeros(self.num_parameters))
        self.register_buffer('n_models', torch.zeros(1, dtype=torch.long))

        # Initialize subspace
        if subspace_kwargs is None:
            subspace_kwargs = dict()
        self.subspace = Subspace.create(subspace_type, num_parameters=self.num_parameters,
                                        **subspace_kwargs)

        self.var_clamp = var_clamp

        self.cov_factor = None
        self.model_device = 'cpu'

    # dont put subspace on cuda?
    def cuda(self, device=None):
        self.model_device = 'cuda'
        self.base_model.cuda(device=device)

    def to(self, *args, **kwargs):
        self.base_model.to(*args, **kwargs)
        device, dtype, non_blocking = torch._C._nn._parse_to(*args, **kwargs)
        self.model_device = device.type
        self.subspace.to(device=torch.device('cpu'), dtype=dtype, non_blocking=non_blocking)

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def collect_model(self, base_model, *args, **kwargs):
        # need to refit the space after collecting a new model
        self.cov_factor = None

        # w = flatten([param.detach().cpu() for param in base_model.parameters()])
        w = flatten([param.detach() for param in base_model.parameters()])
        # first moment
        self.mean.mul_(self.n_models.item() / (self.n_models.item() + 1.0))
        self.mean.add_(w / (self.n_models.item() + 1.0))

        # second moment
        self.sq_mean.mul_(self.n_models.item() / (self.n_models.item() + 1.0))
        self.sq_mean.add_(w ** 2 / (self.n_models.item() + 1.0))

        dev_vector = w - self.mean

        self.subspace.collect_vector(dev_vector, *args, **kwargs)
        self.n_models.add_(1)

    def _get_mean_and_variance(self):
        variance = torch.clamp(self.sq_mean - self.mean ** 2, self.var_clamp)
        return self.mean, variance

    def fit(self):
        if self.cov_factor is not None:
            return
        self.cov_factor = self.subspace.get_space()

    def set_swa(self):
        set_weights(self.base_model, self.mean, self.model_device)

    def sample(self, scale=0.5, diag_noise=True):
        self.fit()
        mean, variance = self._get_mean_and_variance()

        eps_low_rank = torch.randn(self.cov_factor.size()[0])
        z = self.cov_factor.t() @ eps_low_rank
        if diag_noise:
            z += variance * torch.randn_like(variance)
        z *= scale ** 0.5
        sample = mean + z

        # apply to parameters
        set_weights(self.base_model, sample, self.model_device)
        return sample

    def get_space(self, export_cov_factor=True):
        mean, variance = self._get_mean_and_variance()
        if not export_cov_factor:
            return mean.clone(), variance.clone()
        else:
            self.fit()
            return mean.clone(), variance.clone(), self.cov_factor.clone()


class RaPP(nn.Module):
    def __init__(self, in_dim):
        super(RaPP, self).__init__()
        self.enc_layer_list = [nn.Linear(in_dim, 4),
                               nn.ReLU(True),
                                nn.Linear(4, 4),
                                nn.ReLU(True),
                                nn.Linear(4, 4),
                                nn.ReLU(True),
                                nn.Linear(4, 2)
                               ]
        self.encoder = nn.Sequential(*self.enc_layer_list)
        self.decoder = nn.Sequential(nn.Linear(2, 4),
                                     nn.ReLU(True),
                                     nn.Linear(4, 4),
                                     nn.ReLU(True),
                                     nn.Linear(4, 4),
                                     nn.ReLU(True),
                                     nn.Linear(4, in_dim))

    def forward(self, x):
        return self.decoder(self.encoder(x))