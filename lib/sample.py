import torch
import torch.nn as nn

class SampleNormal(nn.Module):
    def __init__(self, dim: int, num_samples: int):
        super(SampleNormal, self).__init__()
        self.dim = dim
        self.num_samples = num_samples
    
    def forward(self, x):
        mu, log_sigma = x[..., :self.dim], x[..., self.dim:]

        return self.sample(mu, log_sigma)
    
    def sample(self, mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        # mu: n_batch, ..., n_dim
        # log_sigma: n_batch, ..., n_dim
        # return: n_samples, n_batch, ..., n_dim
        return mu[None, ...] + torch.randn(self.num_samples, *mu.shape) * torch.exp(log_sigma[None, ...])

class SampleUniform(nn.Module):
    def __init__(self, dim: int, num_samples: int):
        super(SampleUniform, self).__init__()
        self.dim = dim
        self.num_samples = num_samples
    
    def forward(self, x):
        lower, upper = x[..., :self.dim], x[..., self.dim:]

        return self.sample(lower, upper)
    
    def sample(self, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
        # lower: n_batch, ..., n_dim
        # upper: n_batch, ..., n_dim
        # return: n_samples, n_batch, ..., n_dim
        return lower[None, ...] + torch.rand(self.num_samples, *lower.shape) * (upper[None, ...] - lower[None, ...])