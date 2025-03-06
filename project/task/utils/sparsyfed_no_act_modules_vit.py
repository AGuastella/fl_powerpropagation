"""
A PyTorch re-implementation of the following notebook:
https://github.com/deepmind/deepmind-research/blob/master/powerpropagation/powerpropagation.ipynb
written by DeepMind.

Adapted from the code available at: https://github.com/mysistinechapel/powerprop by @jjkc33 and @mysistinechapel
"""

from typing import Union

import torch
from torch import nn
import torch.nn.functional as F
from torch.types import _int, _size

from project.task.utils.spectral_norm import SpectralNormHandler


class SparsyFed_no_act_linear(nn.Module):
    """SparsyFed (no activation pruning) Linear module."""

    def __init__(
        self,
        alpha: float,
        sparsity: float,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ):
        super(SparsyFed_no_act_linear, self).__init__()
        self.alpha = alpha
        self.sparsity = sparsity
        self.in_features = in_features
        self.out_features = out_features
        self.b = bias
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if self.b else None
        self.spectral_norm_handler = SpectralNormHandler()

    def __repr__(self):
        return (
            f"SparsyFed_no_act_linear(alpha={self.alpha}, sparsity={self.sparsity},"
            f" in_features={self.in_features},"
            f" out_features={self.out_features}, bias={self.b})"
        )

    def get_weight(self):
        weight = self.weight.detach()
        if self.alpha == 1.0:
            return weight
        elif self.alpha < 0:
            return self.spectral_norm_handler.compute_weight_update(self.weight)
        return torch.sign(weight) * torch.pow(torch.abs(weight), self.alpha)

    def forward(self, inputs, mask=None):
        # Apply the re-parametrisation to `self.weight` using `self.alpha`
        if self.alpha == 1.0:
            weight = self.weight
        elif self.alpha < 0:
            weight = self.spectral_norm_handler.compute_weight_update(self.weight)
        else:
            weight = torch.sign(self.weight) * torch.pow(
                torch.abs(self.weight), self.alpha
            )
        # Apply a mask, if given
        if mask is not None:
            weight *= mask
        # Compute the linear forward pass usign the re-parametrised weight
        return F.linear(input=inputs, weight=weight, bias=self.bias)
