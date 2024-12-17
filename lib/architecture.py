import torch
import torch.nn as nn

class Search(nn.Module):
    def __init__(
            self,
            transition: nn.Module,
            fitness: nn.Module,
            max_depth: int,
            beam_width: int,
            temperature: float = 1.0,
            reuse_parents: bool = False
        ):
        super(Search, self).__init__()

        # search functions
        self.transition = transition
        self.fitness = fitness

        # search parameters
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.temperature = temperature
        self.reuse_parents = reuse_parents

    def forward(self, x):
        return self.search(x)

    def set_temperature(self, temperature):
        self.temperature = temperature

    def search(self, x: torch.Tensor):
        next_states = x[None, ...]

        for _ in range(self.max_depth):
            next_states = self.next_states(next_states)

        # attention on current beam
        y = torch.linalg.vecdot(next_states, torch.softmax(self.fitness(next_states), dim=0), dim=0)

        return y

    def next_states(self, current_states: torch.Tensor):
        # current_states: (num_samples * n_candidates), n_batch, ..., n_dim
        candidates = self.transition(current_states).flatten(0,1)

        if self.reuse_parents:
            candidates = torch.cat([current_states, candidates])

        # candidates_fitness: (max_width * n_candidates), n_batch, ..., 1
        candidates_fitness = self.fitness(candidates).squeeze(-1) / self.temperature

        if len(candidates) <= self.beam_width:
            return candidates

        if self.train:
            target_shape = (self.beam_width, *candidates_fitness.shape[1:])
            probs = torch.softmax(candidates_fitness, dim=0).movedim(0,-1).flatten(0, -2)
            samples = torch.multinomial(probs, self.beam_width)
            top_candidates = samples.movedim(-1, 0).reshape(target_shape)
        else:
            top_candidates = torch.topk(candidates_fitness, self.beam_width, dim=0).indices

        if len(top_candidates.shape) == 1:
            top_candidates = top_candidates[..., None]

        # return self.beam_width, n_batch, ..., n_dims
        return candidates.gather(0, top_candidates[..., None].expand(self.beam_width, *candidates.shape[1:]))

class RandomizedSearch(Search):
    def __init__(
            self,
            transition: nn.Module,
            sample: nn.Module,
            fitness: nn.Module,
            **args,
        ):
        super(RandomizedSearch, self).__init__(
            nn.Sequential(
                transition,
                sample
            ),
            fitness,
            **args
        )