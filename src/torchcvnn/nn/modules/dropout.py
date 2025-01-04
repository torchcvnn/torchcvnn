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


class Dropout(nn.Module):
    r"""
    Applies dropout to zero out values of the inputs

    During training, randomly zeroes some of the elements of the input tensor
    with probability :math:`p` using samples from a Bernouilli distribution. Each
    channel will be zeroed out independently on every forward call.

    Furthermore, the outputs are scaled by a factor of :math:`\frac{1}{1-p}` during
    training. This means that during evaluation the module simply computes an
    identity function.

    Note:
        As for now, with torch 2.1.0 :external:py:class:`torch.nn.Dropout` cannot be applied on
        complex valued tensors. However, our implementation relies on the torch
        implementation by dropout out a tensor of ones used as a mask on the
        input.

    Arguments:
        p: probability of an element to be zeroed.
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        mask = torch.nn.functional.dropout(
            torch.ones(z.shape), self.p, training=self.training
        ).to(z.device)
        return mask * z


class Dropout2d(nn.Module):
    r"""
    Applies dropout to zero out complete channels

    Note:
        As for now, with torch 2.1.0 :external:py:class:`torch.nn.Dropout2d` cannot be applied on
        complex valued tensors. However, our implementation relies on the torch
        implementation by dropout out a tensor of ones used as a mask on the
        input.

    Arguments:
        p: probability of an element to be zeroed.
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        mask = torch.nn.functional.dropout2d(
            torch.ones(z.shape), self.p, training=self.training
        ).to(z.device)
        return mask * z
