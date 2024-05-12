import torch
import torch.nn as nn

class ExpandLinear(nn.Module):
    def __init__(self, dim: int, factor: int):
        super(ExpandLinear, self).__init__()

        self.dim = dim
        self.factor = factor

        self.linear = nn.Linear(dim, dim * factor)
    
    def forward(self, x: torch.Tensor):
        h = self.linear(x)

        return h.reshape(*h.shape[:-1], self.dim, self.factor).movedim(-1, 0)
