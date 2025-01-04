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
import time

# External imports
import torch

# Local imports
import torchcvnn.nn as c_nn
import torchcvnn.nn.modules.batchnorm as bn


def test_inv():
    # Take a positive semi-definite matrix
    B = 10
    X = torch.randn((B, 2, 2))
    XT = torch.transpose(X, 1, 2)
    XXT = torch.bmm(X, XT) + (0.01 * torch.eye(2)).unsqueeze(0)

    # Compute its inverse square root
    inv = bn.inv_2x2(XXT)
    torch_inv = torch.linalg.inv(XXT)

    # And check this is really the inerse of X
    assert torch.allclose(inv, torch_inv)


def test_inv_sqrt():
    # Take a positive semi-definite matrix
    B = 10
    X = torch.randn((B, 2, 2))
    XT = torch.transpose(X, 1, 2)
    XXT = torch.bmm(X, XT) + (0.05 * torch.eye(2)).unsqueeze(0)

    # Compute its inverse square root
    inv_sqrt = bn.inv_sqrt_2x2(XXT)

    # Square it to have the inverse
    inv = torch.bmm(inv_sqrt, inv_sqrt)

    torch_inv = torch.linalg.inv(XXT)

    # And check this is really the inerse of X
    assert torch.allclose(inv, torch_inv)


def time_it(f, N=1000):
    t0 = time.time()
    for i in range(N):
        f()
    t1 = time.time()
    return (t1 - t0) / N


def time_inv_sqrt():
    # Take a positive semi-definite matrix
    B = 10
    X = torch.randn((B, 2, 2))
    XT = torch.transpose(X, 1, 2)
    XXT = torch.bmm(X, XT) + (0.05 * torch.eye(2)).unsqueeze(0)

    # Compute its inverse square root
    el1 = time_it(lambda: bn.slow_inv_sqrt_2x2(XXT))
    el2 = time_it(lambda: bn.inv_sqrt_2x2(XXT))
    print(el1, el2)


def test_batchnorm1d():
    B, C = 20, 16
    m = c_nn.BatchNorm1d(C)

    x = torch.randn((B, C), dtype=torch.complex64)
    output = m(x)

    # Compute the variance/covariance
    xc = output.transpose(0, 1).reshape(C, -1)  # C, B
    mus = xc.mean(axis=-1)  # 16 means
    xc_real = torch.view_as_real(xc)
    covs = bn.batch_cov(xc_real)  # 16 covariances matrices

    # All the mus must be 0's
    # For some reasons, this is not exactly 0
    assert torch.allclose(mus, torch.zeros_like(mus), atol=1e-7)
    # All the covs must be identities
    id_cov = 0.5 * torch.eye(2).tile(C, 1, 1)
    assert torch.allclose(covs, id_cov, atol=1e-3)


def test_batchnorm2d():
    B, C, H, W = 20, 16, 50, 100
    m = c_nn.BatchNorm2d(C)

    x = torch.randn((B, C, H, W), dtype=torch.complex64)
    output = m(x)

    # Compute the variance/covariance
    xc = output.transpose(0, 1).reshape(C, -1)  # C, BxHxW
    mus = xc.mean(axis=-1)  # 16 means
    xc_real = torch.view_as_real(xc)
    covs = bn.batch_cov(xc_real)  # 16 covariances matrices

    # All the mus must be 0's
    # For some reasons, this is not exactly 0
    assert torch.allclose(mus, torch.zeros_like(mus), atol=1e-7)
    # All the covs must be identities
    id_cov = 0.5 * torch.eye(2).tile(C, 1, 1)
    assert torch.allclose(covs, id_cov, atol=1e-3)


if __name__ == "__main__":
    test_inv()
    test_inv_sqrt()
    # time_inv_sqrt()
    test_batchnorm1d()
    test_batchnorm2d()
