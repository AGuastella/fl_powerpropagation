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
            return self.spectral_norm_handler.compute_weight_update(weight)
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


class SparsyFed_no_act_Conv2D(nn.Module):
    """SparsyFed (no activation pruning) Conv2D module."""

    def __init__(
        self,
        alpha: float,
        sparsity: float,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: Union[_size, _int] = 1,
        padding: Union[_size, _int] = 1,
        dilation: Union[_size, _int] = 1,
        groups: _int = 1,
        bias: bool = False,
    ):
        super(SparsyFed_no_act_Conv2D, self).__init__()
        self.alpha = alpha
        self.sparsity = sparsity
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.b = bias
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.spectral_norm_handler = SpectralNormHandler()

    def __repr__(self):
        return (
            f"SparsyFed_no_act_Conv2D(alpha={self.alpha}, sparsity={self.sparsity},"
            f" in_channels={self.in_channels},"
            f" out_channels={self.out_channels}, kernel_size={self.kernel_size},"
            f" bias={self.b}, stride={self.stride}, padding={self.padding},"
            f" dilation={self.dilation}, groups={self.groups})"
        )

    def get_weights(self):
        weights = self.weight.detach()
        if self.alpha == 1.0:
            return weights
        elif self.alpha < 0:
            return self.spectral_norm_handler.compute_weight_update(weights)
        return torch.sign(weights) * torch.pow(torch.abs(weights), self.alpha)

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
        # Compute the conv2d forward pass usign the re-parametrised weight
        return F.conv2d(
            input=inputs,
            weight=weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class SparsyFed_no_act_Conv1D(nn.Module):
    """SparsyFed (no activation pruning) Conv1D module."""

    def __init__(
        self,
        alpha: float,
        sparsity: float,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: Union[_size, _int] = 1,
        padding: Union[_size, _int] = 1,
        dilation: Union[_size, _int] = 1,
        groups: _int = 1,
        bias: bool = False,
    ):
        super(SparsyFed_no_act_Conv1D, self).__init__()
        self.alpha = alpha
        self.sparsity = sparsity
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.b = bias
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.spectral_norm_handler = SpectralNormHandler()

    def __repr__(self):
        return (
            f"SparsyFed_no_act_Conv1D(alpha={self.alpha}, sparsity={self.sparsity},"
            f" in_channels={self.in_channels},"
            f" out_channels={self.out_channels}, kernel_size={self.kernel_size},"
            f" bias={self.b}, stride={self.stride}, padding={self.padding},"
            f" dilation={self.dilation}, groups={self.groups})"
        )

    def get_weights(self):
        weights = self.weight.detach()
        if self.alpha == 1.0:
            return weights
        return torch.sign(weights) * torch.pow(torch.abs(weights), self.alpha)

    def forward(self, inputs, mask=None):
        # Apply the re-parametrisation to `self.weight` using `self.alpha`
        if self.alpha == 1.0:
            weight = self.weight
        else:
            weight = torch.sign(self.weight) * torch.pow(
                torch.abs(self.weight), self.alpha
            )
        # Apply a mask, if given
        if mask is not None:
            weight *= mask

        return F.conv1d(
            input=inputs,
            weight=weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
