"""This file contains utilities for initializing complex neural network parameters."""

from typing import Optional as _Optional

import torch
from torch import Tensor
from torch.nn.init import _no_grad_trunc_normal_


__all__ = ["trunc_normal_"]


def trunc_normal_(
    tensor: Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    a: float = -2.0,
    b: float = 2.0,
    generator: _Optional[torch.Generator] = None,
) -> Tensor:
    r"""
    see :func:`torch.nn.init.trunc_normal_`

    Fill the input Tensor with values drawn from a truncated normal distribution.

    The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    If the input Tensor is of a complex dtype, apply the same truncated normal
    initialization indpendently on real and imaginary parts.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
        generator: the torch Generator to sample from (default: None)


    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    if tensor.dtype.is_complex:
        _no_grad_trunc_normal_(tensor.real, mean, std, a, b, generator=generator)
        _no_grad_trunc_normal_(tensor.imag, mean, std, a, b, generator=generator)

    else:
        _no_grad_trunc_normal_(tensor, mean, std, a, b, generator=generator)

    return tensor
