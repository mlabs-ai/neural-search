import torch
import torch.nn as nn

from lib.sample import Sampler

class QuantumSearch(nn.Module):
    def __init__(
            self,
            transition: nn.Module,
            fitness: nn.Module,
            max_depth: int,
            branching_width: int,
            beam_width: int
        ):
        super(QuantumSearch, self).__init__()

        # search functions
        self.transition = transition
        self.fitness = fitness

        # search parameters
        self.max_depth = max_depth
        self.branching_width = branching_width
        self.beam_width = beam_width

    def forward(self, x):
        return self.search(x)

    def search(self, x: torch.Tensor):
        beam = x[None, ...]

        for _ in range(self.max_depth):
            beam = self.next_states(beam)

        y = beam.mean(0)

        return y

    def next_states(self, beam: torch.Tensor) -> torch.Tensor:
        # current_states: (branching_width * n_candidates), n_batch, ..., n_dim
        candidates = self.transition(beam, self.branching_width).flatten(0, 1)

        # candidates_fitness: beam_width, (branching_width * n_candidates), n_batch, ..., (n_dim or 1)
        candidates_fitness = self.fitness(candidates, self.beam_width)

        # next_candidates: beam_width, n_batch, ..., n_dim
        next_beam = torch.sum(torch.softmax(candidates_fitness, dim=1) * candidates, dim=1)

        return next_beam

class SamplingOneToManyNetwork(nn.Module):
    def __init__(
            self,
            encoder: nn.Module,
            sampler: Sampler,
            decoder: nn.Module
        ):
        super(SamplingOneToManyNetwork, self).__init__()

        self.encoder: nn.Module = encoder
        self.sampler: Sampler = sampler
        self.decoder: nn.Module = decoder

    def forward(self, x: torch.Tensor, k: int):
        h_dist = self.encoder(x)

        h_samples = self.sampler(h_dist, k)

        return self.decoder(h_samples)

class DeterministicFitnessAttention(nn.Module):
    def __init__(self, dim, beam_width):
        super(DeterministicFitnessAttention, self).__init__()
        self.heads = nn.Parameter(torch.randn(dim, beam_width))

    def forward(self, x: torch.Tensor, k=None):
        scores = x @ self.heads

        return scores.movedim(-1, 0)[..., None]

class SamplingFitnessAttention(nn.Module):
    def __init__(self, dim, sampler: Sampler):
        super(DeterministicFitnessAttention, self).__init__()
        self.dist = nn.Parameter(torch.rand(2 * dim))
        self.sampler: Sampler = sampler

    def forward(self, x: torch.Tensor, k: int):
        heads = self.sampler(x, k)
        scores = x @ heads

        return scores.movedim(-1, 0)[..., None]
