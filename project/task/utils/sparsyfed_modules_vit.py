"""
TODO:
"""

from copy import deepcopy
from logging import log
import logging
from typing import Union
from matplotlib import pyplot as plt
import numpy as np


import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.grad import conv2d_input, conv2d_weight
from torch.nn.modules.utils import _pair
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.types import _int, _size
from project.fed.utils.utils import (
    get_tensor_sparsity,
    nonzeros_tensor,
    print_nonzeros_tensor,
)

from project.task.utils.drop import (
    drop_nhwc_send_th,
    drop_structured,
    drop_structured_filter,
    drop_threshold,
    matrix_drop,
)
from project.task.utils.spectral_norm import SpectralNormHandler

torch.autograd.set_detect_anomaly(True)


# def convolution_backward(
#     ctx,
#     grad_output,
# ):
#     sparse_input, sparse_weight, bias = ctx.saved_tensors
#     conf = ctx.conf
#     input_grad = (
#         weight_grad
#     ) = (
#         bias_grad
#     ) = (
#         sparsity_grad
#     ) = (
#         grad_in_th
#     ) = grad_wt_th = stride_grad = padding_grad = dilation_grad = groups_grad = None

#     # Compute gradient w.r.t. input
#     if ctx.needs_input_grad[0]:
#         input_grad = conv2d_input(
#             sparse_input.shape,
#             sparse_weight,
#             grad_output,
#             conf["stride"],
#             conf["padding"],
#             conf["dilation"],
#             conf["groups"],
#         )

#     # Compute gradient w.r.t. weight
#     if ctx.needs_input_grad[1]:
#         weight_grad = conv2d_weight(
#             sparse_input,
#             sparse_weight.shape,
#             grad_output,
#             conf["stride"],
#             conf["padding"],
#             conf["dilation"],
#             conf["groups"],
#         )

#     # Compute gradient w.r.t. bias (works for every Conv2d shape)
#     if bias is not None and ctx.needs_input_grad[2]:
#         bias_grad = grad_output.sum(dim=(0, 2, 3))

#     return (
#         input_grad,
#         weight_grad,
#         bias_grad,
#         sparsity_grad,
#         grad_in_th,
#         grad_wt_th,
#         stride_grad,
#         padding_grad,
#         dilation_grad,
#         groups_grad,
#     )


# class sparsyfed_linear(Function):
#     @staticmethod
#     def forward(ctx, input, weight, bias, sparsity):

#         if input.dim() == 2 and bias is not None:
#             # The fused op is marginally faster
#             output = torch.addmm(bias, input, weight.t())
#         else:
#             output = input.matmul(weight.t())
#             if bias is not None:
#                 output += bias

#         topk = 1 - sparsity

#         sparse_input = matrix_drop(input, topk)

#         ctx.save_for_backward(sparse_input, weight, bias)
#         return output

#     @staticmethod
#     def backward(ctx, grad_output):
#         sparse_input, sparse_weight, bias = ctx.saved_tensors

#         grad_input = grad_weight = grad_bias = None

#         if ctx.needs_input_grad[0]:
#             grad_input = grad_output.mm(sparse_weight)
#         if ctx.needs_input_grad[1]:
#             grad_weight = grad_output.t().mm(sparse_input)
#         if bias is not None and ctx.needs_input_grad[2]:
#             grad_bias = grad_output.sum(0)

#         return grad_input, grad_weight, grad_bias, None


# class SparsyFedLinear(nn.Module):
#     """Powerpropagation Linear module."""

#     def __init__(
#         self,
#         alpha: float,
#         in_features: int,
#         out_features: int,
#         bias: bool = True,
#         sparsity: float = 0.3,
#     ):
#         super(SparsyFedLinear, self).__init__()
#         self.alpha = alpha
#         self.in_features = in_features
#         self.out_features = out_features
#         self.b = bias
#         self.weight = nn.Parameter(torch.empty(out_features, in_features))
#         self.bias = nn.Parameter(torch.empty(out_features)) if self.b else None
#         self.spectral_norm_handler = SpectralNormHandler()
#         self.sparsity = sparsity

#     def __repr__(self):
#         return (
#             f"SparsyFedLinear(alpha={self.alpha}, in_features={self.in_features},"
#             f" out_features={self.out_features}, bias={self.b},"
#             f" sparsity={self.sparsity})"
#         )

#     def get_weights(self):
#         weights = self.weight.detach()
#         if self.alpha == 1.0:
#             return weights
#         elif self.alpha < 0:
#             return self.spectral_norm_handler.compute_weight_update(weights)
#         return torch.sign(weights) * torch.pow(torch.abs(weights), self.alpha)

#     def _call_sparsyfed_linear(self, input, weight) -> torch.Tensor:
#         if self.training:
#             sparsity = get_tensor_sparsity(weight)
#         else:
#             # Avoid to sparsify during the evaluation
#             sparsity = 0.0
#         return sparsyfed_linear.apply(input, weight, self.bias, sparsity)

#     def forward(self, input):
#         # Apply the re-parametrisation to `self.weight` using `self.alpha`
#         if self.alpha == 1.0:
#             sparsyfed_weight = self.weight
#         elif self.alpha < 0:
#             sparsyfed_weight = self.spectral_norm_handler.compute_weight_update(
#                 self.weight
#             )
#         else:
#             sparsyfed_weight = torch.sign(self.weight) * torch.pow(
#                 torch.abs(self.weight), self.alpha
#             )

#         output = self._call_sparsyfed_linear(input, sparsyfed_weight)

#         # Return the output
#         return output



class sparsyfed_linear(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, sparsity):
        # Store original dtypes and shapes
        ctx.input_dtype = input.dtype
        ctx.weight_dtype = weight.dtype
        ctx.input_shape = input.shape
        
        # Ensure shapes are compatible
        input_reshaped = input
        if input.dim() > 2:
            # For ViT: (B, N, D) -> (B*N, D)
            B = input.size(0)
            input_reshaped = input.reshape(-1, input.size(-1))
        
        # Convert weight to input dtype for forward pass
        weight = weight.to(input.dtype)
        if bias is not None:
            bias = bias.to(input.dtype)
            
        output = F.linear(input_reshaped, weight, bias)
        
        # Reshape output back if needed
        if input.dim() > 2:
            output = output.reshape(B, -1, output.size(-1))
        
        # Compute sparse input using input's dtype
        topk = max(1 - sparsity, 0.01)
        sparse_input = matrix_drop(input_reshaped.contiguous(), topk)
        
        ctx.save_for_backward(sparse_input, weight, bias)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        sparse_input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # Reshape grad_output if needed
        grad_reshaped = grad_output
        if grad_output.dim() > 2:
            grad_reshaped = grad_output.reshape(-1, grad_output.size(-1))

        if ctx.needs_input_grad[0]:
            # Compute grad_input
            weight_for_grad = weight.to(grad_output.dtype)
            grad_input = F.linear(grad_reshaped, weight_for_grad.t())
            # Reshape back to original shape if needed
            if len(ctx.input_shape) > 2:
                grad_input = grad_input.reshape(ctx.input_shape)

        if ctx.needs_input_grad[1]:
            # Convert sparse_input to grad_output dtype
            sparse_input = sparse_input.to(grad_output.dtype)
            # Compute grad_weight
            grad_weight = grad_reshaped.t().mm(sparse_input)
            grad_weight = grad_weight.to(ctx.weight_dtype)

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_reshaped.sum(0)
            grad_bias = grad_bias.to(ctx.weight_dtype)

        return grad_input, grad_weight, grad_bias, None

class SparsyFedLinear(nn.Module):
    """SWAT Linear module adapted for Vision Transformer with numerical stability improvements."""
    
    def __init__(
        self,
        alpha: float,
        in_features: int,
        out_features: int,
        bias: bool = True,
        sparsity: float = 0.3,
        min_abs_value: float = 1e-7,  # Add minimum absolute value threshold
    ):
        super().__init__()
        self.alpha = alpha
        self.in_features = in_features
        self.out_features = out_features
        self.sparsity = sparsity
        self.min_abs_value = min_abs_value
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.spectral_norm_handler = SpectralNormHandler()
        
    def extra_repr(self) -> str:
        return (f'in_features={self.in_features}, '
                f'out_features={self.out_features}, '
                f'bias={self.bias is not None}, '
                f'alpha={self.alpha}, '
                f'sparsity={self.sparsity}, '
                f'min_abs_value={self.min_abs_value}')

    def forward(self, input):
        # Add debug logging for weight statistics
        if torch.isnan(self.weight).any():
            logging.warning("NaN values detected in weights before processing")
            
        if self.alpha == 1.0:
            weight = self.weight
        elif self.alpha < 0:
            weight = self.spectral_norm_handler.compute_weight_update(self.weight)
        else:
            # Apply minimum threshold to prevent very small values
            weight_abs = torch.abs(self.weight)
            weight_abs = torch.clamp(weight_abs, min=self.min_abs_value)
            
            # Safe power operation with clamped values
            powered_weights = torch.pow(weight_abs, self.alpha)
            
            # Restore signs
            weight = torch.sign(self.weight) * powered_weights
            
            # Add debug checks
            if torch.isnan(weight).any():
                logging.error(f"NaN values detected after power operation. Alpha: {self.alpha}")
                logging.error(f"Min weight_abs: {weight_abs.min()}")
                logging.error(f"Max weight_abs: {weight_abs.max()}")
                # Replace NaN values with original weights
                weight = torch.where(torch.isnan(weight), self.weight, weight)
        
        # Final safety check
        weight = torch.nan_to_num(weight, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return sparsyfed_linear.apply(input, weight, self.bias, self.sparsity)


