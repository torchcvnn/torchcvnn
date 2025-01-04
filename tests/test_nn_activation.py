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

# External imports
import torch

# Local imports
import torchcvnn.nn as c_nn


def test_crelu():
    activation = c_nn.CReLU()

    x = torch.randn((3, 3), dtype=torch.complex64)
    output = activation(x)


def test_cprelu():
    activation = c_nn.CPReLU()

    x = torch.randn((3, 3), dtype=torch.complex64)
    output = activation(x)


def test_celu():
    activation = c_nn.CELU()

    x = torch.randn((3, 3), dtype=torch.complex64)
    output = activation(x)


def test_ccelu():
    activation = c_nn.CCELU()

    x = torch.randn((3, 3), dtype=torch.complex64)
    output = activation(x)


def test_cgelu():
    activation = c_nn.CGELU()

    x = torch.randn((3, 3), dtype=torch.complex64)
    output = activation(x)


def test_csigmoid():
    activation = c_nn.CSigmoid()

    x = torch.randn((3, 3), dtype=torch.complex64)
    output = activation(x)


def test_ctanh():
    activation = c_nn.CTanh()

    x = torch.randn((3, 3), dtype=torch.complex64)
    output = activation(x)


def test_zrelu():
    activation = c_nn.zReLU()

    x = torch.randn((3, 3), dtype=torch.complex64)
    output = activation(x)


def test_zabsrelu():
    activation = c_nn.zAbsReLU()

    x = torch.randn((3, 3), dtype=torch.complex64)
    output = activation(x)


def test_zleakyrelu():
    activation = c_nn.zLeakyReLU()

    x = torch.randn((3, 3), dtype=torch.complex64)
    output = activation(x)


def test_mod():
    activation = c_nn.Mod()

    x = torch.randn((3, 3), dtype=torch.complex64)
    output = activation(x)


def test_modReLU():
    activation = c_nn.modReLU()

    x = torch.randn((3, 3), dtype=torch.complex64)
    output = activation(x)


def test_cardioid():
    activation = c_nn.Cardioid()

    x = torch.randn((3, 3), dtype=torch.complex64)
    output = activation(x)


def test_multiheadattention():
    batch_size = 16
    source_seq_len = 9
    target_seq_len = 11

    embed_dim = 12
    kdim = 8
    vdim = 9
    num_heads = 3

    for batch_first in [True, False]:

        mha = c_nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=batch_first, kdim=kdim, vdim=vdim
        )

        query = torch.randn(
            (
                (batch_size, target_seq_len, embed_dim)
                if batch_first
                else (target_seq_len, batch_size, embed_dim)
            ),
            dtype=torch.complex64,
        )
        key = torch.randn(
            (
                (batch_size, source_seq_len, kdim)
                if batch_first
                else (source_seq_len, batch_size, kdim)
            ),
            dtype=torch.complex64,
        )
        value = torch.randn(
            (
                (batch_size, source_seq_len, vdim)
                if batch_first
                else (source_seq_len, batch_size, vdim)
            ),
            dtype=torch.complex64,
        )

        # Check for the forward pass
        # Ensure it runs without exception
        output = mha(query, key, value)

        expected_output_shape = (
            [batch_size, target_seq_len, embed_dim]
            if batch_first
            else [target_seq_len, batch_size, embed_dim]
        )

        # Check for the shape
        assert list(output[0].shape) == expected_output_shape


if __name__ == "__main__":
    test_crelu()
    test_cprelu()
    test_celu()
    test_ccelu()
    test_cgelu()
    test_csigmoid()
    test_ctanh()
    test_zrelu()
    test_zabsrelu()
    test_zleakyrelu()
    test_mod()
    test_modReLU()
    test_cardioid()
    test_multiheadattention()
