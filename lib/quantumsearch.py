from typing import Union
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
        next_beam = torch.sum(torch.softmax(candidates_fitness, dim=1) * candidates, dim=1) + beam
          # next_candidates: beam_width, n_batch, ..., n_dim
        # next_beam = torch.sum(torch.softmax(candidates_fitness, dim=1) * candidates, dim=1) + beam


        return next_beam


""" Regularized Quantum Search. """

class RegularizedQuantumSearch(nn.Module):
    def __init__(
            self,
            transition: nn.Module,
            fitness: nn.Module,
            max_depth: int,
            branching_width: int,
            beam_width: int,
           ):
        super(RegularizedQuantumSearch, self).__init__()

        # search functions
        self.transition = transition
        self.fitness = fitness

        # search parameters
        self.max_depth = max_depth
        self.branching_width = branching_width
        self.beam_width = beam_width
        self._entropy =None


    def forward(self, x):
        # reset entropy
        self._entropy = None
        return self.search(x)

    def search(self, x: torch.Tensor):
        beam = x[None, ...]
        entropies = []


        for _ in range(self.max_depth):
            beam, entropy = self.next_states(beam)
            entropies.append(entropy.mean())
        

        y = beam.mean(0)
        self._entropy = torch.mean(torch.stack(entropies))
        return y

    def next_states(self, beam: torch.Tensor) -> torch.Tensor:
        # current_states: (branching_width * n_candidates), n_batch, ..., n_dim
        candidates = self.transition(beam, self.branching_width).flatten(0, 1)

        # candidates_fitness: beam_width, (branching_width * n_candidates), n_batch, ..., (n_dim or 1)
        candidates_fitness = self.fitness(candidates, self.beam_width)
        probs = torch.softmax(candidates_fitness, dim=1)

        #calculate entropy
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim =1)

        # next_candidates: beam_width, n_batch, ..., n_dim
        next_beam = torch.sum(probs * candidates, dim=1) + beam

        return next_beam, entropy

    def get_entropy(self) -> float:
        if self._entropy is None:
           return torch.tensor(0.0)
        return self._entropy


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

class OneToManyNetwork(nn.Module):
    def __init__(self, network: nn.Module):
        super(OneToManyNetwork, self).__init__()
        self.network = network

    def forward(self, x: torch.Tensor, k: int = None):
        return self.network(x)

class FitnessFunction(nn.Module):
    def __init__(self, one_to_many: Union[SamplingOneToManyNetwork, OneToManyNetwork]):
        super(FitnessFunction, self).__init__()
        self.one_to_many = one_to_many

    def forward(self, x: torch.Tensor, k: int):
        heads = self.one_to_many(x, k)

        return heads.movedim(-1, 0)

class TransitionFunction(nn.Module):
    def __init__(self, one_to_many: Union[SamplingOneToManyNetwork, OneToManyNetwork]):
        super(TransitionFunction, self).__init__()
        self.one_to_many = one_to_many

    def forward(self, x: torch.Tensor, k: int):
        heads = self.one_to_many(x, k)

        return heads.movedim(-1, 0)
