import math
from typing import List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from uncertainty_modeling.rl_uncertainty.rank1.set_encoder import LatentPerturber
# from utils import ConvArgs


def convr1_bn_relu(
    ch: int, filters: int, p: Optional[float], pool: str = None
) -> List[nn.Module]:
    """
    VGG 13, this model was referenced from the deep ensemble paper which tehn referenced this site
    http://torch.ch/blog/2015/07/30/cifar.html
    """

    if pool is not None and not (pool == "max" or pool == "avg"):
        raise ValueError(f"pool must be max avg or None. got: {pool}")

    out = [
        Conv2dRank1(ch, filters, 3, 1, padding=1),
        nn.BatchNorm2d(filters),
        nn.LeakyReLU(),
    ]

    if pool == "avg":
        out.append(nn.AvgPool2d(2, padding=0))
    elif pool == "max":
        out.append(nn.MaxPool2d(2, padding=0))

    if p is not None:
        out.append(nn.Dropout(p))

    return out


class Gaussian(object):
    def __init__(self, mu: torch.Tensor, rho: torch.Tensor) -> None:
        super().__init__()
        self.mu = mu
        self.rho = rho
        self.normal = torch.distributions.Normal(0, 1)

    @property
    def sigma(self) -> torch.Tensor:
        return torch.log1p(torch.exp(self.rho))

    def sample(self, device: torch.device) -> torch.Tensor:
        epsilon = self.normal.sample(self.rho.size()).to(device)
        return self.mu + self.sigma * epsilon  # type: ignore


class RankOneBayesianVector(nn.Module):
    def __init__(self, n: int, ft: int) -> None:
        super().__init__()
        self.n = n
        self.ft = ft

        # Weight parameters
        self.weight_mu = nn.Parameter(
            torch.empty(n, ft).uniform_(-0.2, 0.2), requires_grad=True
        )
        self.weight_rho = nn.Parameter(
            torch.empty(n, ft).uniform_(-5, -4), requires_grad=True
        )
        self.weight = Gaussian(self.weight_mu, self.weight_rho)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight.sample(device=x.device)

        self.kl = -0.5 * torch.sum(
            1
            + 2 * torch.log(self.weight.sigma)
            - self.weight_mu ** 2
            - self.weight.sigma ** 2
        )

        # x = (n, batch, ft) * (n, ft, 1) = elementwise in the feature dimension -> (n, batch, 1)
        return x * weight.unsqueeze(1)


class RankOneBayesianVectorConv(nn.Module):
    def __init__(self, n: int, ft: int) -> None:
        super().__init__()
        self.n = n
        self.ft = ft

        # Weight parameters
        self.weight_mu = nn.Parameter(
            torch.empty(n, ft).uniform_(-0.2, 0.2), requires_grad=True
        )
        self.weight_rho = nn.Parameter(
            torch.empty(n, ft).uniform_(-5, -4), requires_grad=True
        )
        self.weight = Gaussian(self.weight_mu, self.weight_rho)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight.sample(device=x.device)
        weight = weight.view(weight.size(0), 1, weight.size(1), 1, 1)

        self.kl = -0.5 * torch.sum(
            1
            + 2 * torch.log(self.weight.sigma)
            - self.weight_mu ** 2
            - self.weight.sigma ** 2
        )

        # x = (n, batch, ft) * (n, ft, 1) = elementwise in the feature dimension -> (n, batch, 1)
        return x * weight


class Linear(nn.Module):
    def __init__(self, in_ft: int, out_ft: int) -> None:
        super(Linear, self).__init__()
        self.in_features = in_ft
        self.out_features = out_ft
        self.weight = nn.Parameter(torch.empty(out_ft, in_ft))
        self.bias = nn.Parameter(torch.empty(out_ft))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (n, batch, in) *w.T.unsqueeze: (1, in, out) -> (n, batch, out)
        h = torch.matmul(x, self.weight.T.unsqueeze(0))
        return h + self.bias.unsqueeze(0).unsqueeze(0)


class Model(nn.Module):
    def __init__(self, x_dim: int, h_dim: int, y_dim: int, n: int) -> None:
        super(Model, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.out_dim = y_dim
        self.n = n

        self.l1_s = RankOneBayesianVector(n, x_dim)
        self.l1_shared = Linear(x_dim, h_dim)
        self.l1_r = RankOneBayesianVector(n, h_dim)

        self.l2_s = RankOneBayesianVector(n, h_dim)
        self.l2_shared = Linear(h_dim, h_dim)
        self.l2_r = RankOneBayesianVector(n, h_dim)

        self.mu_s = RankOneBayesianVector(n, h_dim)
        self.mu_shared = Linear(h_dim, y_dim)
        self.mu_r = RankOneBayesianVector(n, y_dim)

        self.logvar_s = RankOneBayesianVector(n, h_dim)
        self.logvar_shared = Linear(h_dim, y_dim)
        self.logvar_r = RankOneBayesianVector(n, y_dim)

    def kl(self) -> torch.Tensor:
        return (
            self.l1_s.kl
            + self.l1_r.kl
            + self.l2_s.kl
            + self.l2_r.kl
            + self.mu_s.kl
            + self.mu_r.kl
            + self.logvar_s.kl
            + self.logvar_r.kl
        ) / 8

    def forward(  # type: ignore
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # if this is during testing, it will be the same points, so we need to repeat it
        if len(x.size()) == 2:
            x = x.repeat(self.n, 1, 1)
        # print(x.size())
        h = self.l1_s(x)
        h = self.l1_shared(h)
        h = F.relu(self.l1_r(h))

        h = self.l2_s(h)
        h = self.l2_shared(h)
        h = F.relu(self.l2_r(h))

        mu = self.mu_s(h)
        mu = self.mu_shared(mu)
        mu = self.mu_r(mu)

        logvar = self.logvar_s(h)
        logvar = self.logvar_shared(logvar)
        logvar = self.logvar_r(logvar)
        # print(mu.size(), logvar.size())
        return mu.squeeze(2) # , logvar.squeeze(2)

    def mc(self, x: torch.Tensor, samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        mus = torch.zeros(self.n, samples, x.size(0), device=x.device)
        logvars = torch.zeros(self.n, samples, x.size(0), device=x.device)
        for i in range(samples):
            mus[:, i], logvars[:, i] = self(x)

        return mus, logvars


class Conv2dRank1(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        model_n: int = 5,
        padding_mode: str = "zeros",  # TODO: refine this type
    ) -> None:
        super(Conv2dRank1, self).__init__()

        self.conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

        # s should be (model_n, input channel)
        self.s_vector = RankOneBayesianVectorConv(model_n, in_channels)
        # r should be (model_n, filters)
        self.r_vector = RankOneBayesianVectorConv(model_n, out_channels)

        self.model_n = model_n
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print("x: ", x.size())
        # print(f"x: {x.view(self.model_n, -1, *x.size()[1:]).size()}")
        x = self.s_vector(x.view(self.model_n, -1, *x.size()[1:]))
        # print(f"after s x is: {x.size()}")

        mdl_n, b, ch, h, w = x.size()

        x = x.view(mdl_n * b, ch, h, w)
        # print(f"after resize: {x.size()}")

        x = self.conv2d(x)

        # print(f"after conv: {x.size()}")
        x = x.view(mdl_n, b, *x.size()[1:])
        # print(f"x after resize: {x.size()}")

        x = self.r_vector(x)
        # print(f"x after r: {x.size()}")
        x = x.view(-1, self.out_channels, w, h)
        # print(f"return size: {x.size()}")

        return x

'''
class ModelConv(nn.Module):
    def __init__(
        self,
        x_dim: Tuple[int, int],
        h_dim: int,
        y_dim: int,
        n: int,
        conv_args: ConvArgs,
    ) -> None:
        super(ModelConv, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.y_dim = y_dim
        self.n = n

        conv_layers = []
        for arg in conv_args:
            conv_layers += convr1_bn_relu(*arg)

        self.kl_layers = []
        for i, layer in enumerate(conv_layers):
            if isinstance(layer, Conv2dRank1):
                self.kl_layers.append(i)

        self.layers = nn.Sequential(*conv_layers)

        self.l1_s_vector = RankOneBayesianVector(n, conv_args[-1][1])
        self.l1_shared = Linear(conv_args[-1][1], h_dim)
        self.l1_r_vector = RankOneBayesianVector(n, h_dim)

        self.l2_s_vector = RankOneBayesianVector(n, h_dim)
        self.l2_shared = Linear(h_dim, h_dim)
        self.l2_r_vector = RankOneBayesianVector(n, h_dim)

        self.out_s_vector = RankOneBayesianVector(n, h_dim)
        self.out_shared = Linear(h_dim, y_dim)
        self.out_r_vector = RankOneBayesianVector(n, y_dim)

    def zero_kl(self) -> None:
        for i in self.kl_layers:
            self.layers[i].s_vector.kl = None  # type: ignore
            self.layers[i].r_vector.kl = None  # type: ignore

        self.l1_s_vector.kl = None  # type: ignore
        self.l1_r_vector.kl = None  # type: ignore
        self.l2_s_vector.kl = None  # type: ignore
        self.l2_r_vector.kl = None  # type: ignore
        self.out_s_vector.kl = None  # type: ignore
        self.out_r_vector.kl = None  # type: ignore

    def kl(self) -> torch.Tensor:
        kl = 0.0
        for i in self.kl_layers:
            kl += self.layers[i].s_vector.kl + self.layers[i].r_vector.kl  # type: ignore

        return (
            kl
            + self.l1_s_vector.kl
            + self.l1_r_vector.kl
            + self.l2_s_vector.kl
            + self.l2_r_vector.kl
            + self.out_s_vector.kl
            + self.out_r_vector.kl
        ) / (6 + 2 * len(self.kl_layers))

    def forward(self, x: torch.Tensor, val: bool = False) -> torch.Tensor:
        if val:
            # if this is during testing, it will be the same points going to each model, so we need to repeat it
            x = x.repeat(self.n, 1, 1, 1, 1)
            x = x.view(-1, *x.size()[2:])

        h = self.layers(x)

        h = h.view(self.n, -1, *h.size()[1:]).squeeze(-1).squeeze(-1)
        return self.classifier(h)

    def classifier(self, x: torch.Tensor) -> torch.Tensor:
        x = self.l1_s_vector(x)
        x = self.l1_shared(x)
        x = F.leaky_relu(self.l1_r_vector(x))

        x = self.l2_s_vector(x)
        x = self.l2_shared(x)
        x = F.leaky_relu(self.l2_r_vector(x))

        x = self.out_s_vector(x)
        x = self.out_shared(x)
        x = self.out_r_vector(x)

        return x

    def phi(
        self, x: torch.Tensor, perturber: LatentPerturber, theta: bool = True
    ) -> Tuple[torch.Tensor, ...]:

        h = self.layers(x).squeeze(-1).squeeze(-1)
        # print(f"h after layers: {h.size()}")

        h_prime, entropy, subsets = perturber(h)
        dist = ((h.unsqueeze(0) - h_prime.unsqueeze(1)) ** 2).sum(dim=2)

        # print(f"h' after perturb: {h_prime.size()} dist: {dist.size()}")

        h_prime = h_prime.view(self.n, -1, *h_prime.size()[1:])

        if theta:
            # we only want x_prime to apply to the top layers of the network, as in regression when we
            # only operate on the input space.
            h_prime = h_prime.detach()
            h_prime.requires_grad_(True)

        # print(f"h' after view: {h_prime.size()}")
        h_prime = self.classifier(h_prime)

        return h_prime, entropy, subsets, dist

    def mc(self, x: torch.Tensor, samples: int) -> torch.Tensor:
        preds = torch.zeros(samples, x.size(0), self.y_dim, device=x.device)
        for i in range(samples):
            preds[i] = self(x, val=True).mean(dim=0)

        return preds
'''
