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
import math

# External imports
import torch
import torch.nn as nn
import torch.nn.init as init


def batch_cov(points: torch.Tensor, centered: bool = False) -> torch.Tensor:
    """
    Batched covariance computation
    Adapted from : https://stackoverflow.com/a/71357620/2164582

    Arguments:
        points: the (B, N, D) input tensor from which to compute B covariances
        centered: If `True`, assumes for every batch, the N vectors are centered.  default: `False`

    Returns:
        bcov: the covariances as a `(B, D, D)` tensor
    """
    B, N, D = points.size()
    if not centered:
        mean = points.mean(dim=1).unsqueeze(1)
        diffs = (points - mean).reshape(B * N, D)
    else:
        diffs = points.reshape(B * N, D)
    prods = torch.bmm(diffs.unsqueeze(2), diffs.unsqueeze(1)).reshape(B, N, D, D)
    bcov = prods.sum(dim=1) / (N - 1)  # Unbiased estimate
    return bcov  # (B, D, D)


def inv_2x2(M: torch.Tensor) -> torch.Tensor:
    r"""
    Computes the inverse of a tensor of shape [N, 2, 2].

    If we denote

    .. math::

        M = \begin{pmatrix} a & b \\ c & d \end{pmatrix}

    The inverse is given by

    .. math::

        M^{-1} = \frac{1}{Det M} Adj(M) = \frac{1}{ad - bc}\begin{pmatrix}d & -b \\ -c & a\end{pmatrix}

    Arguments:
        M: a batch of 2x2 tensors to invert, i.e. a :math:`(B, 2, 2)` tensor
    """
    det = torch.linalg.det(M).unsqueeze(-1).unsqueeze(-1)

    M_adj = M.clone()
    M_adj[:, 0, 0], M_adj[:, 1, 1] = M[:, 1, 1], M[:, 0, 0]
    M_adj[:, 0, 1] *= -1
    M_adj[:, 1, 0] *= -1
    M_inv = 1 / det * M_adj
    return M_inv


def sqrt_2x2(M: torch.Tensor) -> torch.Tensor:
    r"""
    Computes the square root of the tensor of shape [N, 2, 2].

    If we denote

    .. math::

        M = \begin{pmatrix} a & b \\ c & d \end{pmatrix}

    The square root is given by :

    .. math::

        \begin{align}
        \sqrt{M} &= \frac{1}{t} ( M + \sqrt{Det M} I)\\
        t &= \sqrt{Tr M + 2 \sqrt{Det M}}
        \end{align}

    Arguments:
        M: a batch of 2x2 tensors to invert, i.e. a :math:`(B, 2, 2)` tensor
    """
    N = M.shape[0]
    det = torch.linalg.det(M).unsqueeze(-1).unsqueeze(-1)
    sqrt_det = torch.sqrt(det)

    trace = torch.diagonal(M, dim1=-2, dim2=-1).sum(-1).unsqueeze(-1).unsqueeze(-1)
    t = torch.sqrt(trace + 2 * sqrt_det)

    sqrt_M = 1 / t * (M + sqrt_det * torch.eye(2, device=M.device).tile(N, 1, 1))
    return sqrt_M


def slow_inv_sqrt_2x2(M: torch.Tensor) -> torch.Tensor:
    """
    Computes the square root of the inverse of a tensor of shape [N, 2, 2]

    Arguments:
        M: a batch of 2x2 tensors to sqrt invert, i.e. a :math:`(B, 2, 2)` tensor
    """
    return sqrt_2x2(inv_2x2(M))


def inv_sqrt_2x2(M: torch.Tensor) -> torch.Tensor:
    """
    Computes the square root of the inverse of a tensor of shape [N, 2, 2]

    Arguments:
        M: a batch of 2x2 tensors to sqrt invert, i.e. a :math:`(B, 2, 2)` tensor
    """
    N = M.shape[0]
    det = torch.linalg.det(M).unsqueeze(-1).unsqueeze(-1)
    sqrt_det = torch.sqrt(det)

    trace = torch.diagonal(M, dim1=-2, dim2=-1).sum(-1).unsqueeze(-1).unsqueeze(-1)
    t = torch.sqrt(trace + 2 * sqrt_det)

    M_adj = M.clone()
    M_adj[:, 0, 0], M_adj[:, 1, 1] = M[:, 1, 1], M[:, 0, 0]
    M_adj[:, 0, 1] *= -1
    M_adj[:, 1, 0] *= -1
    M_sqrt_inv = (
        1 / t * (M_adj / sqrt_det + torch.eye(2, device=M.device).tile(N, 1, 1))
    )
    return M_sqrt_inv


class _BatchNormNd(nn.Module):
    r"""
    BatchNorm for complex valued neural networks. The same code applies for
    BatchNorm1d, BatchNorm2d, the only condition being the input tensor must be
    (batch_size, features, d1, d2, ..)

    The statistics will be computed over the :math:`batch\_size \times d_1 \times d_2 \times ..`
    vectors of size `features`.

    As defined by Trabelsi et al. (2018)

    Arguments:
        num_features: :math:`C` from an expected input of size :math:`(B, C)`
        eps: a value added to the denominator for numerical stability. Default :math:`1e-5`.
        momentum: the value used for the running mean and running var computation. Can be set to `None` for cumulative moving average (i.e. simple average). Default: :math:`0.1`
        affine: a boolean value that when set to `True`, this module has learnable affine parameters. Default: `True`
        track_running_stats: a boolean value that when set to `True`, this module tracks the running mean and variance, and when set to`False`, this module does not track such statistics, and initializes statistics buffers running_mean and running_var as None. When these buffers are None, this module always uses batch statistics. in both training and eval modes. Default: `True`
        cdtype: the dtype for complex numbers. Default torch.complex64
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device: torch.device = None,
        cdtype: torch.dtype = torch.complex64,
    ) -> None:
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = torch.nn.parameter.Parameter(
                torch.empty((num_features, 2, 2), device=device)
            )
            self.bias = torch.nn.parameter.Parameter(
                torch.empty((num_features,), device=device, dtype=cdtype)
            )
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        if self.track_running_stats:
            # Register the running mean and running variance
            # These will not be returned by model.parameters(), hence
            # not updated by the optimizer although returned in the state_dict
            # and therefore stored as model's assets
            self.register_buffer(
                "running_mean",
                torch.zeros((num_features,), device=device, dtype=cdtype),
            )
            self.register_buffer(
                "running_var", torch.ones((num_features, 2, 2), device=device)
            )
            self.register_buffer(
                "num_batches_tracked", torch.tensor(0, dtype=torch.long, device=device)
            )
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer(
                "num_batches_tracked",
                None,
            )
        self.reset_parameters()

    def reset_running_stats(self) -> None:
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.zero_()
            self.running_var[:, 0, 0] = 1 / math.sqrt(2.0)
            self.running_var[:, 1, 1] = 1 / math.sqrt(2.0)

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.reset_running_stats()
            if self.affine:
                # Initialize all the weights to zeros
                init.zeros_(self.weight)
                # And then fill in the diagonal with 1/sqrt(2)
                # w is C, 2, 2
                self.weight[:, 0, 0] = 1 / math.sqrt(2.0)
                self.weight[:, 1, 1] = 1 / math.sqrt(2.0)
                # Initialize all the biases to zero
                init.zeros_(self.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z : [B, C, d1, d2, ..] (complex)
        batch_size = z.shape[0]
        dim1 = z.shape[1]
        other_dims = z.shape[2:]

        xc = z.transpose(0, 1).reshape(self.num_features, -1)  # num_features, BxHxW

        if self.training or not self.track_running_stats:
            # For training
            # Or for testing but using the batch stats for centering/scaling

            # Compute the means
            mus = xc.mean(axis=-1)  # num_features means

            # Center the xc
            xc_centered = xc - mus.unsqueeze(-1)  # num_features, BxHxW
            xc_centered = torch.view_as_real(xc_centered)  # num_features, BxHxW, 2

            # Transform the complex numbers as 2 reals to compute the variances and
            # covariances
            covs = batch_cov(xc_centered, centered=True)  # 16 covariances matrices
        else:
            # The means come from the running stats
            mus = self.running_mean

            # Center the xc
            xc_centered = xc - mus.unsqueeze(-1)  # num_features, BxHxW
            xc_centered = torch.view_as_real(xc_centered)  # num_features, BxHxW, 2

            # The variance/covariance come from the running stats
            covs = self.running_var

        # Invert the covariance to scale
        invsqrt_covs = inv_sqrt_2x2(
            covs + self.eps * torch.eye(2, device=covs.device)
        )  # num_features, 2, 2
        # Note: the xc_centered.transpose is to make
        # xc_centered from (C, BxHxW, 2) to (B, 2, BxHxW)
        # So that the batch matrix multiply works as expected
        # where invsqrt_covs is (C, 2, 2)
        outz = torch.bmm(invsqrt_covs, xc_centered.transpose(1, 2))
        outz = outz.contiguous()  # num_features, 2, BxHxW

        # Shift by beta and scale by gamma
        # weight is (num_features, 2, 2) real valued
        outz = torch.bmm(self.weight, outz)  # num_features, 2, BxHxW
        outz = outz.transpose(1, 2).contiguous()
        outz = torch.view_as_complex(outz)  # num_features, BxHxW

        # bias is (C, ) complex dtype
        outz += self.bias.view((self.num_features, 1))

        # With the following operation, weight
        # outz = outz.reshape(C, B, H, W).transpose(0, 1)
        outz = outz.reshape(dim1, batch_size, *other_dims).transpose(0, 1)

        if self.training and self.track_running_stats:
            self.running_mean = (
                1.0 - self.momentum
            ) * self.running_mean + self.momentum * mus
            if torch.isnan(self.running_mean).any():
                raise RuntimeError("Running mean divergence")

            self.running_var = (
                1.0 - self.momentum
            ) * self.running_var + self.momentum * covs
            if torch.isnan(self.running_var).any():
                raise RuntimeError("Running var divergence")
        return outz


class BatchNorm1d(_BatchNormNd):
    r"""
    BatchNorm for complex valued neural networks. The same code applies for
    BatchNorm1d, BatchNorm2d, the only condition being the input tensor must be
    (batch_size, features, d1, d2, ..)

    The statistics will be computed over the :math:`batch\_size \times d_1 \times d_2 \times ..`
    vectors of size `features`.

    As defined by Trabelsi et al. (2018)

    Arguments:
        num_features: :math:`C` from an expected input of size :math:`(B, C)`
        eps: a value added to the denominator for numerical stability. Default :math:`1e-5`.
        momentum: the value used for the running mean and running var computation. Can be set to `None` for cumulative moving average (i.e. simple average). Default: :math:`0.1`
        affine: a boolean value that when set to `True`, this module has learnable affine parameters. Default: `True`
        track_running_stats: a boolean value that when set to `True`, this module tracks the running mean and variance, and when set to`False`, this module does not track such statistics, and initializes statistics buffers running_mean and running_var as None. When these buffers are None, this module always uses batch statistics. in both training and eval modes. Default: `True`
        cdtype: the dtype for complex numbers. Default torch.complex64
    """

    pass


class BatchNorm2d(_BatchNormNd):
    r"""
    BatchNorm for complex valued neural networks. The same code applies for
    BatchNorm1d, BatchNorm2d, the only condition being the input tensor must be
    (batch_size, features, d1, d2, ..)

    The statistics will be computed over the :math:`batch\_size \times d_1 \times d_2 \times ..`
    vectors of size `features`.

    As defined by Trabelsi et al. (2018)

    Arguments:
        num_features: :math:`C` from an expected input of size :math:`(B, C)`
        eps: a value added to the denominator for numerical stability. Default :math:`1e-5`.
        momentum: the value used for the running mean and running var computation. Can be set to `None` for cumulative moving average (i.e. simple average). Default: :math:`0.1`
        affine: a boolean value that when set to `True`, this module has learnable affine parameters. Default: `True`
        track_running_stats: a boolean value that when set to `True`, this module tracks the running mean and variance, and when set to`False`, this module does not track such statistics, and initializes statistics buffers running_mean and running_var as None. When these buffers are None, this module always uses batch statistics. in both training and eval modes. Default: `True`
        cdtype: the dtype for complex numbers. Default torch.complex64
    """

    pass
