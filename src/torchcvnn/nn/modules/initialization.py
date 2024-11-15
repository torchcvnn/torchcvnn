# MIT License

# Copyright (c) 2024 Quentin Gabot

# The following is an adaptation of the initialization code from
# PyTorch library for the complex-valued neural networks

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
import math
import warnings

# External imports
import torch
import torch.nn as nn


def complex_kaiming_normal_(
    tensor: torch.Tensor,
    a: float = 0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
) -> torch.Tensor:
    r"""Fills the input `Tensor` with values according to the method
    described in `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification` - He, K. et al. (2015), using a
    normal distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{N}(0, \text{std}^2)` where

    .. math::
        \text{std} = \frac{\text{gain}}{\sqrt{2\text{fan_mode}}}

    Also known as He initialization.

    Arguments:
        tensor: an n-dimensional `torch.Tensor`
        a: the negative slope of the rectifier used after this layer (only
            used with ``'leaky_relu'``)
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
        nonlinearity: the non-linear function (`nn.functional` name),
            recommended to use only with ``'relu'`` or ``'leaky_relu'`` (default).

    Examples:
        >>> w = torch.empty(3, 5, dtype=torch.complex64)
        >>> c_nn.init.complex_kaiming_normal_(w, mode='fan_out', nonlinearity='relu')

    This implementation is a minor adaptation of the :external:py:func:`torch.nn.init.kaiming_normal_` function
    """

    fan = nn.init._calculate_correct_fan(tensor, mode)
    gain = nn.init.calculate_gain(nonlinearity, a)
    std = (gain / math.sqrt(fan)) / math.sqrt(2)

    return nn.init._no_grad_normal_(tensor, 0.0, std)


def complex_kaiming_uniform_(
    tensor: torch.Tensor,
    a: float = 0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
) -> torch.Tensor:
    r"""Fills the input `Tensor` with values according to the method
    described in `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification` - He, K. et al. (2015), using a
    uniform distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-\text{bound}, \text{bound})` where

    .. math::

        \text{bound} = \text{gain} \times \sqrt{\frac{3}{2\text{fan_mode}}}

    Also known as He initialization.

    Arguments:
        tensor: an n-dimensional :external:py:class`torch.Tensor`
        a: the negative slope of the rectifier used after this layer (only
            used with ``'leaky_relu'``)
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
        nonlinearity: the non-linear function (`nn.functional` name),
            recommended to use only with ``'relu'`` or ``'leaky_relu'`` (default).

    Examples:
        >>> w = torch.empty(3, 5, dtype=torch.complex64)
        >>> c_nn.init.complex_kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')

    This implementation is a minor adaptation of the :external:py:func:`torch.nn.init.kaiming_uniform_` function
    """

    if 0 in tensor.shape:
        warnings.warn("Initializing zero-element tensors is a no-op")
        return tensor
    fan = nn.init._calculate_correct_fan(tensor, mode)
    gain = nn.init.calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan) / math.sqrt(2)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation

    return nn.init._no_grad_uniform_(tensor, -bound, bound)


def complex_xavier_uniform_(
    tensor: torch.Tensor,
    a: float = 0,
    nonlinearity: str = "leaky_relu",
) -> torch.Tensor:
    r"""Fills the input `Tensor` with values according to the method
    described in `Understanding the difficulty of training deep feedforward
    neural networks` - Glorot, X. & Bengio, Y. (2010), using a uniform
    distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-a, a)` where

    .. math::

        a = \text{gain} \times \sqrt{\frac{6}{2(\text{fan_in} + \text{fan_out})}}

    Also known as Glorot initialization.

    Arguments:
        tensor: an n-dimensional `torch.Tensor`
        a: an optional parameter to the non-linear function
        nonlinearity: the non linearity to compute the gain

    Examples:
        >>> w = torch.empty(3, 5, dtype=torch.complex64)
        >>> c_nn.init.complex_xavier_uniform_(w, nonlinearity='relu')

    This implementation is a minor adaptation of the :external:py:func:`torch.nn.init.xavier_uniform_` function
    """
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    gain = nn.init.calculate_gain(nonlinearity, a)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out)) / math.sqrt(2)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation

    return nn.init._no_grad_uniform_(tensor, -bound, bound)


def complex_xavier_normal_(
    tensor: torch.Tensor,
    a: float = 0,
    nonlinearity: str = "leaky_relu",
) -> torch.Tensor:
    r"""Fills the input `Tensor` with values according to the method
    described in `Understanding the difficulty of training deep feedforward
    neural networks` - Glorot, X. & Bengio, Y. (2010), using a normal
    distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{N}(0, \text{std}^2)` where

    .. math::

        \text{std} = \text{gain} \times \sqrt{\frac{2}{2(\text{fan_in} + \text{fan_out})}}

    Also known as Glorot initialization.

    Arguments:
        tensor: an n-dimensional `torch.Tensor`
        a: an optional parameter to the non-linear function
        nonlinearity: the non linearity to compute the gain

    Examples:
        >>> w = torch.empty(3, 5, dtype=torch.complex64)
        >>> nn.init.complex_xavier_normal_(w, nonlinearity='relu')

    This implementation is a minor adaptation of the :external:py:func:`torch.nn.init.xavier_normal_` function
    """
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    gain = nn.init.calculate_gain(nonlinearity, a)
    std = (gain * math.sqrt(2.0 / float(fan_in + fan_out))) / math.sqrt(2)

    return nn.init._no_grad_normal_(tensor, 0.0, std)
