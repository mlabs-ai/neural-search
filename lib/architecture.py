from typing import Tuple
import torch
import torch.nn as nn

class Search(nn.Module):
    def __init__(
            self, 
            transition: nn.Module, 
            fitness: nn.Module,
            normalization: nn.Module,
            max_depth: int,
            max_width: int,
            beam_width: int
        ):
        super(Search, self).__init__()
        # search functions
        self.transition = transition
        self.fitness = fitness

        # stability
        self.normalization = normalization

        # search parameters
        self.max_depth = max_depth
        self.max_width = max_width
        self.beam_width = beam_width
    
    def forward(self, x):
        return self.search(x)
    
    def search(self, x: torch.Tensor):
        next_states = x[None, ...]

        for _ in range(self.max_depth):
            next_states = self.next_states(next_states)

        # attention on current beam
        y = torch.linalg.vecdot(next_states, torch.softmax(self.fitness(next_states), dim=0), dim=0)

        return y

    def next_states(self, current_states: torch.Tensor):
        # current_states: n_candidates, n_batch, ..., n_dim
        next_state_distributions = self.transition(current_states)

        next_state_samples = self.sample(*next_state_distributions, self.max_width).flatten(0, 1)

        # candidates: (max_width * n_candidates), n_batch, ..., n_dim
        candidates = self.normalization(next_state_samples)

        # candidates_fitness: (max_width * n_candidates), n_batch, ..., n_dim
        candidates_fitness = self.fitness(candidates).squeeze()

        if self.train:
            top_candidates = torch.multinomial(torch.softmax(candidates_fitness.T, dim=-1), self.beam_width).T
        else:
            top_candidates = torch.topk(candidates_fitness, self.beam_width, dim=0).indices

        # return self.beam_width, n_batch, ..., n_dims
        return candidates.gather(0, top_candidates[..., None].expand(self.beam_width, *candidates.shape[1:]))
    
    def sample(self, mu: torch.Tensor, log_sigma: torch.Tensor, number_of_samples: int) -> torch.Tensor:
        # mu: n_batch, ..., n_dim
        # log_sigma: n_batch, ..., n_dim
        # return: n_samples, n_batch, ..., n_dim
        return mu[None, ...] + torch.randn(number_of_samples, *mu.shape) * torch.exp(log_sigma[None, ...])
