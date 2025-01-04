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
import time

# External imports
import torch

# Local imports
import torchcvnn.nn as c_nn
import torchcvnn.nn.modules.batchnorm as bn


def test_layernorm_real():
    # NLP Example
    batch, sentence_length, embedding_dim = 20, 5, 10
    embedding = torch.randn(batch, sentence_length, embedding_dim)
    layer_norm = torch.nn.LayerNorm(embedding_dim)
    layer_norm.eval()
    # Activate module
    out = layer_norm(embedding)

    assert out.shape == embedding.shape

    out = out.view(batch, sentence_length, -1)  # batch, sentence_length, embeddig_dim

    mus = out.mean(axis=-1)
    assert torch.allclose(mus, torch.zeros_like(mus), atol=1e-6)
    stds = out.std(axis=-1)
    assert torch.allclose(stds, torch.ones_like(stds), atol=1e-1)

    # Image Example
    N, C, H, W = 20, 5, 10, 10
    x = torch.randn(N, C, H, W)
    # Normalize over the last three dimensions (i.e. the channel and spatial dimensions)
    # as shown in the image below
    layer_norm = torch.nn.LayerNorm([C, H, W])
    layer_norm.eval()
    out = layer_norm(x)

    out = out.view(N, -1)  # N, C*H*W

    mus = out.mean(axis=-1)
    assert torch.allclose(mus, torch.zeros_like(mus), atol=1e-6)
    stds = out.std(axis=-1)
    assert torch.allclose(stds, torch.ones_like(stds), atol=1e-1)


def test_layernorm():
    #############
    # NLP Example
    batch, sentence_length, embedding_dim = 20, 5, 10
    embedding = torch.randn(
        (batch, sentence_length, embedding_dim), dtype=torch.complex64
    )
    layer_norm = c_nn.LayerNorm(embedding_dim)
    layer_norm.eval()
    # For the unit test, we randomly initialize the bias
    # to avoid testing only the particular case bias = 0
    torch.nn.init.normal_(layer_norm.bias)

    # Activate module
    out = layer_norm(embedding)

    assert out.shape == embedding.shape  # batch x sentence_length x embedding_dim

    # The out tensor is considered as a collection of batch x sentence_length
    # vectors of size embedding_dim
    out = out.view(-1, embedding_dim).transpose(0, 1)

    mus = out.mean(axis=-1)
    # Check the means equal zeros. Note that, in general, we expect
    assert torch.allclose(mus, layer_norm.bias, atol=1e-6)

    out_real = torch.view_as_real(out)
    covs = bn.batch_cov(out_real)
    id_cov = 0.5 * torch.eye(2).tile(embedding_dim, 1, 1)

    assert torch.allclose(covs, id_cov, atol=1e-3)

    ###############
    # Image Example
    N, C, H, W = 20, 5, 10, 10
    embedding_dim = C * H * W
    x = torch.randn((N, C, H, W), dtype=torch.complex64)

    # Normalize over the last three dimensions (i.e. the channel and spatial dimensions)
    # as shown in the image below
    layer_norm = c_nn.LayerNorm([C, H, W])
    layer_norm.eval()
    # For the unit test, we randomly initialize the bias
    # to avoid testing only the particular case bias = 0
    torch.nn.init.normal_(layer_norm.bias)

    # Activate module
    out = layer_norm(x)

    assert out.shape == x.shape  # N, C, H, W

    # The out tensor is considered as a collection of N
    # vectors of size embedding_dim
    out = out.view(-1, embedding_dim).transpose(0, 1)

    mus = out.mean(axis=-1)
    # Check the means equal zeros. Note that, in general, we expect
    assert torch.allclose(mus, layer_norm.bias, atol=1e-6)

    out_real = torch.view_as_real(out)
    covs = bn.batch_cov(out_real)
    id_cov = 0.5 * torch.eye(2).tile(embedding_dim, 1, 1)

    assert torch.allclose(covs, id_cov, atol=1e-3)


def test_rmsnorm():
    #############
    # NLP Example
    batch, sentence_length, embedding_dim = 20, 5, 10
    embedding = torch.randn(
        (batch, sentence_length, embedding_dim), dtype=torch.complex64
    )
    rms_norm = c_nn.RMSNorm(embedding_dim)
    rms_norm.eval()

    # Activate module
    out = rms_norm(embedding)

    assert out.shape == embedding.shape  # batch x sentence_length x embedding_dim

    ###############
    # Image Example
    N, C, H, W = 20, 5, 10, 10
    embedding_dim = C * H * W
    x = torch.randn((N, C, H, W), dtype=torch.complex64)

    # Normalize over the last three dimensions (i.e. the channel and spatial dimensions)
    # as shown in the image below
    rms_norm = c_nn.RMSNorm([C, H, W])
    rms_norm.eval()

    # Activate module
    out = rms_norm(x)

    assert out.shape == x.shape  # N, C, H, W


if __name__ == "__main__":
    test_layernorm_real()
    test_layernorm()
    test_rmsnorm()
