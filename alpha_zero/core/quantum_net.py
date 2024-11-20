# Copyright (c) 2023 Michael Hu.
# This code is part of the book "The Art of Reinforcement Learning: Fundamentals, Mathematics, and Implementation with Python.".
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


"""AlphaZero Neural Network component."""
import math
from typing import NamedTuple, Tuple
import torch
from torch import nn
import torch.nn.functional as F

from lib.layers import Residual, UnpackGrid, MultiBatchConv2d
from lib.quantumsearch import FitnessFunction, OneToManyNetwork, QuantumSearch
from lib.quantumsearch import TransitionFunction


class NetworkOutputs(NamedTuple):
    pi_prob: torch.Tensor
    value: torch.Tensor


def calc_conv2d_output(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """takes a tuple of (h,w) and returns a tuple of (h,w)"""

    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size, kernel_size)
    h = math.floor(((h_w[0] + (2 * pad) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1)
    w = math.floor(((h_w[1] + (2 * pad) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1)
    return h, w


def initialize_weights(net: nn.Module) -> None:
    """Initialize weights for Conv2d and Linear layers using kaming initializer."""
    assert isinstance(net, nn.Module)

    for module in net.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')

            if module.bias is not None:
                nn.init.zeros_(module.bias)


class ResNetBlock(nn.Module):
    """Basic redisual block."""

    def __init__(
        self,
        num_input_filters: int,
        num_output_filters: int,
        input_size: int,
        residual: bool =  True,

       ) -> None:
        super().__init__()
        self.residual = residual
        self.input_size = input_size

        self.conv_block1 = nn.Sequential(
            MultiBatchConv2d(
                in_channels = num_input_filters,
                out_channels = num_input_filters,
                kernel_size = 3,
                stride = 1,
                padding = 1,
                bias = False,
            ),
            # nn.BatchNorm2d(num_features=num_filters),

           )


        self.conv_block2 = nn.Sequential(
            MultiBatchConv2d(
                in_channels = num_input_filters,
                out_channels = num_output_filters,
                kernel_size = 3,
                stride = 1,
                padding = 1,
                bias = False,
            ),
            # nn.BatchNorm2d(num_features=num_filters),
        )
        if self.residual:
            self.conv_block3 = MultiBatchConv2d(
                    in_channels = num_input_filters,
                    out_channels = num_output_filters,
                    kernel_size = 1,
                    stride = 1,
                    bias = False,
                )
        self.layer_norm1 =  nn.LayerNorm([num_input_filters, self.input_size[0], self.input_size[1]])
        self.layer_norm2 = nn.LayerNorm([num_output_filters, self.input_size[0], self.input_size[1]])

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        out = self.conv_block1(x)
        _,_,C,H,W = out.shape
        # if self.layer_norm1 is None:
        #     self.layer_norm1 = nn.LayerNorm([C, H, W]).to(out.device)
        out = self.layer_norm1(out)
        out = F.relu(out)
        out = self.conv_block2(out)
        # _,_,C,H,W = out.shape
        # if self.layer_norm2 is None:
        #     self.layer_norm2 = nn.LayerNorm([C, H, W]).to(out.device)

        out = self.layer_norm2(out)
        if self.residual:
            residual = self.conv_block3(x)
            out += residual
        out = F.relu(out)
        return out

class UnpackedResidual(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        h = self.module(x)
        return h + x[..., None]

class QuantumAlphaZeroNet(nn.Module):
    """Policy network for AlphaZero agent."""

    def __init__(
        self,
        input_shape: Tuple,
        num_actions: int,
        num_filters: int = 32,
        max_depth: int = 10,
        branching_width: int = 3,
        beam_width: int = 3,
        num_fc_units: int = 256,
        num_search: int =1,
        gomoku: bool = False,
    ) -> None:
        super().__init__()
        c, h, w = input_shape

        # We need to use additional padding for Gomoku to fix agent shortsighted on edge cases
        num_padding = 3 if gomoku else 1

        conv_out_hw = calc_conv2d_output((h, w), 3, 1, num_padding)
        # FIX BUG, Python 3.7 has no math.prod()
        conv_out = conv_out_hw[0] * conv_out_hw[1]

        # First convolutional block
        self.conv_block = nn.Sequential(
            MultiBatchConv2d(
                in_channels=c,
                out_channels=num_filters,
                kernel_size=3,
                stride=1,
                padding=num_padding,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=num_filters),
            nn.ReLU(),
        )


        self.search = nn.Sequential(*list(QuantumSearch(
            transition =  TransitionFunction(
                OneToManyNetwork(
                     UnpackedResidual(nn.Sequential(
                        ResNetBlock(num_input_filters=num_filters, num_output_filters = branching_width* num_filters, input_size = conv_out_hw, residual = False),
                        UnpackGrid(branching_width) # Batch, ...,  3 * H -> Batch, ..., H, 3
                    ))
                ),
            ),
            fitness=FitnessFunction(
                OneToManyNetwork(
                    nn.Sequential(

                        ResNetBlock(num_input_filters=num_filters, num_output_filters = beam_width, input_size = conv_out_hw),
                        UnpackGrid(beam_width) # Batch, ...,  3 * H -> Batch, ..., 1, 3
                    )
                ),
            ),
            max_depth = max_depth,
            beam_width = beam_width,
            branching_width = branching_width)
            for _ in range(num_search)
        ))


        self.policy_head = nn.Sequential(
            nn.Conv2d(
                in_channels = num_filters,
                out_channels=2,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=2),

            nn.ReLU(),
            nn.Flatten(1),
            nn.Linear(2 * conv_out, num_actions),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=1,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=1),

            nn.ReLU(),
            nn.Flatten(1),
            nn.Linear(1 * conv_out, num_fc_units),
            nn.ReLU(),
            nn.Linear(num_fc_units, 1),
            nn.Tanh(),
        )

        initialize_weights(self)

    def forward(self, x: torch.Tensor) -> NetworkOutputs:
        """Given raw state x, predict the raw logits probability distribution for all actions,
        and the evaluated value, all from current player's perspective."""

        out = self.conv_block(x)
        out = self.search(out)

        # Predict raw logits distributions wrt policy
        pi_logits = self.policy_head(out)

        # Predict evaluated value from current player's perspective.
        value = self.value_head(out)

        return pi_logits, value
