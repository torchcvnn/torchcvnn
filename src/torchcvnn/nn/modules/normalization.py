# MIT License

# Copyright (c) 2024 Jeremy Fix

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Standard imports
from typing import Union, List
import math
import numbers
import functools
import operator

# External imports
import torch
from torch import Size
import torch.nn as nn
import torch.nn.init as init

# Local imports
import torchcvnn.nn.modules.batchnorm as bn

_shape_t = Union[int, List[int], Size]


class LayerNorm(nn.Module):
    r"""
    Implementation of the torch.nn.LayerNorm for complex numbers.

    Arguments:
        normalized_shape (int or list or torch.Size): input shape from an expected input of size :math:`(*, normalized\_shape[0], normalized\_shape[1], ..., normalized\_shape[-1])`
        eps (float) – a value added to the denominator for numerical stability. Default: 1e-5
        elementwise_affine (bool): a boolean value that when set to `True`, this module has learnable per-element affine parameters initialized to a diagonal matrix with diagonal element :math:`\frac{1}{\sqrt{2}}` (for weights) and zeros (for biases). Default: `True`
        bias (bool): if set to `False`, the layer will not learn an additive bias
    """

    def __init__(
        self,
        normalized_shape: _shape_t,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = torch.complex64,
    ) -> None:
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps

        self.elementwise_affine = elementwise_affine

        self.combined_dimensions = functools.reduce(operator.mul, self.normalized_shape)
        if self.elementwise_affine:
            self.weight = torch.nn.parameter.Parameter(
                torch.empty((self.combined_dimensions, 2, 2), device=device)
            )
            if bias:
                self.bias = torch.nn.parameter.Parameter(
                    torch.empty((self.combined_dimensions,), device=device, dtype=dtype)
                )
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        r"""
        Initialize the weight and bias. The weight is initialized to a diagonal
        matrix with diagonal :math:`\frac{1}{\sqrt{2}}`.

        The bias is initialized to :math:`0`.
        """
        with torch.no_grad():
            if self.elementwise_affine:
                # Initialize all the weights to zeros
                init.zeros_(self.weight)
                # And then fill in the diagonal with 1/sqrt(2)
                # w is C, 2, 2
                self.weight.view(-1, 2, 2)[:, 0, 0] = 1 / math.sqrt(2.0)
                self.weight.view(-1, 2, 2)[:, 1, 1] = 1 / math.sqrt(2.0)
                # Initialize all the biases to zero
                init.zeros_(self.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass
        """
        # z: *, normalized_shape[0] , ..., normalized_shape[-1]
        z_ravel = z.view(-1, self.combined_dimensions).transpose(0, 1)

        # Compute the means
        mus = z_ravel.mean(axis=-1)  # normalized_shape[0]x normalized_shape[1], ...

        # Center the inputs
        z_centered = z_ravel - mus.unsqueeze(-1)
        z_centered = torch.view_as_real(
            z_centered
        )  # combined_dimensions,num_samples, 2

        # Transform the complex numbers as 2 reals to compute the variances and
        # covariances
        covs = bn.batch_cov(z_centered, centered=True)

        # Invert the covariance to scale
        invsqrt_covs = bn.inv_sqrt_2x2(
            covs + self.eps * torch.eye(2, device=covs.device)
        )  # combined_dimensions, 2, 2
        # Note: the z_centered.transpose is to make
        # z_centered from (combined_dimensions, num_samples, 2) to (combined_dimensions, 2, num_samples)
        # So that the batch matrix multiply works as expected
        # where invsqrt_covs is (combined_dimensions, 2, 2)
        outz = torch.bmm(invsqrt_covs, z_centered.transpose(1, 2))
        outz = outz.contiguous()  # num_features, 2, BxHxW

        # Shift by beta and scale by gamma
        # weight is (num_features, 2, 2) real valued
        outz = torch.bmm(self.weight, outz)  # combined_dimensions, 2, num_samples
        outz = outz.transpose(1, 2).contiguous()  # combined_dimensions, num_samples, 2
        outz = torch.view_as_complex(outz)  # combined_dimensions, num_samples

        # bias is (C, ) complex dtype
        outz += self.bias.view(-1, 1)

        outz = outz.transpose(0, 1).contiguous()  # num_samples, comnbined_dimensions

        outz = outz.view(z.shape)

        return outz


class RMSNorm(nn.Module):
    r"""
    Implementation of the torch.nn.RMSNorm for complex numbers.

    Arguments:
        normalized_shape (int or list or torch.Size): input shape from an expected input of size :math:`(*, normalized\_shape[0], normalized\_shape[1], ..., normalized\_shape[-1])`
        eps (float) – a value added to the denominator for numerical stability. Default: 1e-5
        elementwise_affine (bool): a boolean value that when set to `True`, this module has learnable per-element affine parameters initialized to a diagonal matrix with diagonal element :math:`\frac{1}{\sqrt{2}}` (for weights) and zeros (for biases). Default: `True`
    """

    def __init__(
        self,
        normalized_shape: _shape_t,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = torch.complex64,
    ) -> None:
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps

        self.elementwise_affine = elementwise_affine

        self.combined_dimensions = functools.reduce(operator.mul, self.normalized_shape)
        if self.elementwise_affine:
            self.weight = torch.nn.parameter.Parameter(
                torch.empty((self.combined_dimensions, 2, 2), device=device)
            )
        else:
            self.register_parameter("weight", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        r"""
        Initialize the weights. The weight is initialized to a diagonal
        matrix with diagonal :math:`\frac{1}{\sqrt{2}}`.
        """
        with torch.no_grad():
            if self.elementwise_affine:
                # Initialize all the weights to zeros
                init.zeros_(self.weight)
                # And then fill in the diagonal with 1/sqrt(2)
                # w is C, 2, 2
                self.weight.view(-1, 2, 2)[:, 0, 0] = 1 / math.sqrt(2.0)
                self.weight.view(-1, 2, 2)[:, 1, 1] = 1 / math.sqrt(2.0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass
        """
        # z: *, normalized_shape[0] , ..., normalized_shape[-1]
        z_ravel = z.view(-1, self.combined_dimensions).transpose(0, 1)

        z_ravel = torch.view_as_real(z_ravel)  # combined_dimensions,num_samples, 2

        # Transform the complex numbers as 2 reals to compute the variances and
        # covariances
        covs = bn.batch_cov(z_ravel, centered=True)

        # Invert the covariance to scale
        invsqrt_covs = bn.inv_sqrt_2x2(
            covs + self.eps * torch.eye(2, device=covs.device)
        )  # combined_dimensions, 2, 2
        # Note: the z_centered.transpose is to make
        # z_centered from (combined_dimensions, num_samples, 2) to (combined_dimensions, 2, num_samples)
        # So that the batch matrix multiply works as expected
        # where invsqrt_covs is (combined_dimensions, 2, 2)
        outz = torch.bmm(invsqrt_covs, z_ravel.transpose(1, 2))
        outz = outz.contiguous()  # num_features, 2, BxHxW

        # Scale by gamma
        # weight is (num_features, 2, 2) real valued
        outz = torch.bmm(self.weight, outz)  # combined_dimensions, 2, num_samples
        outz = outz.transpose(1, 2).contiguous()  # combined_dimensions, num_samples, 2
        outz = torch.view_as_complex(outz)  # combined_dimensions, num_samples

        # bias is (C, ) complex dtype
        outz = outz.transpose(0, 1).contiguous()  # num_samples, comnbined_dimensions

        outz = outz.view(z.shape)

        return outz
