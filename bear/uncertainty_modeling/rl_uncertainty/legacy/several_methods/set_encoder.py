from typing import Tuple

import torch
from torch import nn
from torch.distributions import Normal


class LatentPerturber(nn.Module):
    def __init__(self, in_dim: int) -> None:
        """
        this should be the same as the set encoder, but we have already passed the examples through a
        feature extractor so we don't have to mess around with embedding layers
        """
        super(LatentPerturber, self).__init__()

        self.in_dim = in_dim
        self.decoder = nn.Sequential(
            nn.Linear(in_dim * 2, in_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(in_dim * 2, in_dim * 2),
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, ...]]:
        n = int(torch.randint(1, 5, (1,)).item())

        # perm = torch.argsort(torch.rand(x.size(0), x.size(0), device=x.device), dim=1)
        # subsets = perm[:, :n]

        d = ((x.unsqueeze(0) - x.unsqueeze(1)) ** 2).sum(dim=2)
        subsets = torch.argsort(d, dim=1)[:, :n]

        # create the subset index which will be used to get the subsets distance from the pairwise distance
        # matrix. linspace is needed because it needs to be a tuple of (x, y) coordinates
        # fmt: off
        sub_idx = (
            torch.linspace(0, x.size(0) - 1, x.size(0), device=x.device).repeat(n, 1).T.flatten().long(),  # type: ignore
            subsets.flatten().long()
        )
        # fmt: on

        # print(f"subsets: {subsets.size()} sub idx: {sub_idx.size()}")

        z = x[subsets]

        mu = z.mean(dim=1)
        mx, _ = z.max(dim=1)
        z = torch.cat((mu, mx), dim=1)

        z = self.decoder(z)
        dist = Normal(z[:, : self.in_dim], torch.exp(z[:, self.in_dim :] / 2))
        x = x + dist.rsample()

        return x, dist.entropy().mean(), sub_idx


class SetEncoder(nn.Module):
    def __init__(self, in_dim: int, h_dim: int, out_dim: int) -> None:
        super(SetEncoder, self).__init__()
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.out_dim = out_dim

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, h_dim), nn.ReLU(), nn.Linear(h_dim, h_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(h_dim * 2, h_dim * 2),
            nn.ReLU(),
            nn.Linear(h_dim * 2, out_dim * 2),
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.encoder(x)

        # x is (batch, h_dim)
        # - choose a random number for subsubsets
        n = int(torch.randint(2, min(5, x.size(0)), (1,)).item())
        # n = int(torch.randint(2, x.size(0) // 2, (1,)).item())

        # print(f"x: {x.size()}")
        d = ((x.unsqueeze(0) - x.unsqueeze(1)) ** 2).sum(dim=2)
        # print(d[0])
        d = torch.argsort(d, dim=1)[:, :n]
        # print(f"argmin: {d[:,:n]}")
        # print(d.size())

        # - make random array of indices
        # perm = torch.argsort(torch.rand(x.size(0), x.size(0)), dim=1)
        # perm = perm[:, :n]
        # print(f"perm: {perm.size()} perm: {perm.size()}")
        # raise ValueError()

        # z = x[perm]
        z = x[d]

        mu = z.mean(dim=1)
        mx, _ = z.max(dim=1)
        z = torch.cat((mu, mx), dim=1)

        z = self.decoder(z)

        return z[:, : self.out_dim], z[:, self.out_dim :], d


class MultiplicativeNoiseEncoder(nn.Module):
    def __init__(self, in_dim: int, h_dim: int, out_dim: int) -> None:
        super(MultiplicativeNoiseEncoder, self).__init__()
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.out_dim = out_dim

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, h_dim), nn.ReLU(), nn.Linear(h_dim, h_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(h_dim * 2 + in_dim, h_dim * 2 + in_dim),
            nn.ReLU(),
            nn.Linear(h_dim * 2 + in_dim, out_dim * 2),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)

        z = torch.cat((z.mean(dim=0), z.std(dim=0)), dim=0)
        z = torch.cat((x, z.repeat(x.size(0), 1)), dim=1)
        z = self.decoder(z)

        return 1 + z[:, : self.out_dim], z[:, self.out_dim :]
