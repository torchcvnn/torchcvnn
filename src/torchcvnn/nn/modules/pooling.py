# MIT License

# Copyright (c) 2023 Jérémie Levi, Victor Dhédin, Jeremy Fix

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
from typing import Optional

# External imports
import torch
import torch.nn as nn
from torch.nn.common_types import _size_2_t


class MaxPool2d(nn.Module):
    r"""Applies a 2D max pooling on the module of the input signal

    In the simplest case, the output value of the layer with input size :math:`(N, C, H, W)`, output :math:`(N, C, H_{out}, W_{out})` and `kernel_size` `(kH, kW)` can be precisely described as:

    .. math::

        \begin{aligned}
        out(N_i, C_j, h, w) ={} & \max_{m=0, \ldots, kH-1} \max_{n=0, \ldots, kW-1} \\
                                & \text{|input|}(N_i, C_j, \text{stride[0]} \times h + m,
                                                       \text{stride[1]} \times w + n)
        \end{aligned}

    Internally, it is relying on the :external:py:class:`torch.nn.MaxPool2d`
    
    Arguments:
        kernel_size: thr size of the window to take a max over
        stride: the stride of the window
        padding: implicit negative infinity padding to be added
        dilation: a parameter that controls the stride of elements in the window
        ceil_mode: when `True`, use `ceil` instead of `floor` to compute the output shape
        return_indices: if `True`, will return the max indices along with the outputs

    """

    def __init__(
        self,
        kernel_size: _size_2_t,
        stride: Optional[_size_2_t] = None,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        ceil_mode: bool = False,
        return_indices: bool = False,
    ) -> None:
        super().__init__()
        self.return_indices = return_indices
        self.m = nn.MaxPool2d(
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode=ceil_mode,
            return_indices=True,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Computes and return the MaxPool over the magnitude of the input
        """
        _, indices = self.m(torch.abs(z))

        if self.return_indices:
            return z.flatten()[indices], indices
        else:
            return z.flatten()[indices]


class AvgPool2d(nn.Module):
    """
    Implementation of torch.nn.AvgPool2d for complex numbers.
    Apply AvgPool2d on the real and imaginary part.
    Returns complex values associated to the AvgPool2d results.

    Arguments:
        kernel_size: thr size of the window to compute the average
        stride: the stride of the window
        padding: implicit negative infinity padding to be added
        ceil_mode: when `True`, use `ceil` instead of `floor` to compute the output shape
        count_include_pad: when `True`, will include the zero-padding in the averaging calculation
        divisor_override: if specified, it will be used as divisor, otherwise size of the pooling region will be used.
    """

    def __init__(
        self,
        kernel_size: _size_2_t,
        stride: Optional[_size_2_t] = None,
        padding: _size_2_t = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: Optional[int] = None,
    ) -> None:
        super().__init__()
        if type(kernel_size) == int:
            self.kernel_size = [kernel_size] * 2 + [1]
        elif type(kernel_size) == tuple:
            if len(kernel_size) < 3:
                self.kernel_size = [kernel_size] + [1]
            else:
                self.kernel_size = kernel_size

        if type(stride) == int:
            self.stride = [stride] * 2 + [1]
        elif type(stride) == tuple:
            if len(stride) < 3:
                self.stride = [stride] + [1]
            else:
                self.stride = stride

        if type(padding) == int:
            self.padding = [padding] * 2 + [0]
        elif type(padding) == tuple:
            if len(padding) < 3:
                self.padding = [padding] + [0]
            else:
                self.padding = padding

        self.m = torch.nn.AvgPool3d(
            self.kernel_size,
            self.stride,
            self.padding,
            ceil_mode,
            count_include_pad,
            divisor_override,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Computes the average over the real and imaginery parts.
        """
        return torch.view_as_complex(self.m(torch.view_as_real(z)))
