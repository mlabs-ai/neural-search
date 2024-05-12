import torch
import torch.nn as nn

class Search(nn.Module):
    def __init__(
            self, 
            transition: nn.Module, 
            fitness: nn.Module,
            sample: nn.Module,
            max_depth: int,
            beam_width: int,
            temperature: float = 1.0
        ):
        super(Search, self).__init__()

        # search functions
        self.transition = transition
        self.fitness = fitness
        self.sample = sample

        # search parameters
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.temperature = temperature
    
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
        # current_states: n_candidates, n_batch, ..., n_dim
        next_state_distributions = self.transition(current_states)

        # candidates: (num_samples * n_candidates), n_batch, ..., n_dim
        candidates = self.sample(next_state_distributions).flatten(0, 1)

        # candidates_fitness: (max_width * n_candidates), n_batch, ..., n_dim
        candidates_fitness = self.fitness(candidates).squeeze() / self.temperature

        if self.train:
            top_candidates = torch.multinomial(torch.softmax(candidates_fitness.T, dim=-1), self.beam_width).T
        else:
            top_candidates = torch.topk(candidates_fitness, self.beam_width, dim=0).indices
        
        if len(top_candidates.shape) == 1:
            top_candidates = top_candidates[..., None]

        # return self.beam_width, n_batch, ..., n_dims
        return candidates.gather(0, top_candidates[..., None].expand(self.beam_width, *candidates.shape[1:]))
