import torch
import torch.nn as nn

class Sampler(nn.Module):
    def __init__(self):
        super(Sampler, self).__init__()

class SampleNormal(nn.Module):
    def __init__(self, dim: int, num_samples: int = None):
        super(SampleNormal, self).__init__()
        self.dim = dim
        self.num_samples = num_samples

    def forward(self, x, k=None):
        mu, log_sigma = x[..., :self.dim], x[..., self.dim:]

        return self.sample(mu, log_sigma, k if k is not None else self.num_samples)

    def sample(self, mu: torch.Tensor, log_sigma: torch.Tensor, k: int) -> torch.Tensor:
        # mu: n_batch, ..., n_dim
        # log_sigma: n_batch, ..., n_dim
        # return: n_samples, n_batch, ..., n_dim
        return mu[None, ...] + torch.randn(k, *mu.shape) * torch.exp(log_sigma[None, ...])

class SampleUniform(nn.Module):
    def __init__(self, dim: int, num_samples: int = None):
        super(SampleUniform, self).__init__()
        self.dim = dim
        self.num_samples = num_samples

    def forward(self, x, k=None):
        lower, upper = x[..., :self.dim], x[..., self.dim:]

        return self.sample(lower, upper, k if k is not None else self.num_samples)

    def sample(self, lower: torch.Tensor, upper: torch.Tensor, k: int) -> torch.Tensor:
        # lower: n_batch, ..., n_dim
        # upper: n_batch, ..., n_dim
        # return: n_samples, n_batch, ..., n_dim
        return lower[None, ...] + torch.rand(k, *lower.shape) * (upper[None, ...] - lower[None, ...])