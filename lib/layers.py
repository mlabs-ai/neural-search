import torch
import torch.nn as nn
from math import prod

class ExpandLinear(nn.Module):
    def __init__(self, dim: int, factor: int):
        super(ExpandLinear, self).__init__()

        self.dim = dim
        self.factor = factor

        self.linear = nn.Linear(dim, dim * factor)

    def forward(self, x: torch.Tensor):
        h = self.linear(x)

        return h.reshape(*h.shape[:-1], self.dim, self.factor).movedim(-1, 0)

class Residual(nn.Module):
    def __init__(self, module: nn.Module):
        super(Residual, self).__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)

class Unpack(nn.Module):
    def __init__(self, size):
        super(Unpack, self).__init__()
        self.size = size

    def forward(self, x):
        return x.reshape(*x.shape[:-1], x.shape[-1]//self.size, self.size)

class UnpackGrid(nn.Module):
    def __init__(self, size):
        super(UnpackGrid, self).__init__()
        self.size = size

    def forward(self,x):

        return x.reshape(*x.shape[:-3], x.shape[-3]//3, 3,*x.shape[-2:]).movedim(-3,-1)



class MultiBatchConv2d(nn.Conv2d):


    def forward(self, x):
        original_shape = x.shape

        single_batch_x = x.reshape(prod(original_shape[:-3]) , *original_shape[-3:])

        single_batch_y = super(MultiBatchConv2d, self).forward(single_batch_x)

        multi_batch_y = single_batch_y.reshape(*original_shape[:-3], *single_batch_y.shape[-3:])

        return multi_batch_y