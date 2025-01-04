# MIT License

# Copyright (c) 2025 Jeremy Fix

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
import torch.nn as nn

# Local imports
from . import nn as c_nn


def vit_t(
    patch_embedder: nn.Module,
    device: torch.device = None,
    dtype: torch.dtype = torch.complex64,
):
    factory_kwargs = {"device": device, "dtype": dtype}
    num_layers = 12
    num_heads = 3
    hidden_dim = 192
    mlp_dim = 4 * 192
    dropout = 0.0
    attention_dropout = 0.0
    norm_layer = c_nn.RMSNorm

    return c_nn.ViT(
        patch_embedder,
        num_layers,
        num_heads,
        hidden_dim,
        mlp_dim,
        dropout=dropout,
        attention_dropout=attention_dropout,
        norm_layer=norm_layer,
        **factory_kwargs
    )


def vit_s(
    patch_embedder: nn.Module,
    device: torch.device = None,
    dtype: torch.dtype = torch.complex64,
):
    factory_kwargs = {"device": device, "dtype": dtype}
    num_layers = 12
    num_heads = 6
    hidden_dim = 384
    mlp_dim = 4 * 384
    dropout = 0.0
    attention_dropout = 0.0
    norm_layer = c_nn.RMSNorm

    return c_nn.ViT(
        patch_embedder,
        num_layers,
        num_heads,
        hidden_dim,
        mlp_dim,
        dropout=dropout,
        attention_dropout=attention_dropout,
        norm_layer=norm_layer,
        **factory_kwargs
    )


def vit_b(
    patch_embedder: nn.Module,
    device: torch.device = None,
    dtype: torch.dtype = torch.complex64,
):
    factory_kwargs = {"device": device, "dtype": dtype}
    num_layers = 12
    num_heads = 12
    hidden_dim = 768
    mlp_dim = 3072
    dropout = 0.0
    attention_dropout = 0.0
    norm_layer = c_nn.RMSNorm

    return c_nn.ViT(
        patch_embedder,
        num_layers,
        num_heads,
        hidden_dim,
        mlp_dim,
        dropout=dropout,
        attention_dropout=attention_dropout,
        norm_layer=norm_layer,
        **factory_kwargs
    )


def vit_l(
    patch_embedder: nn.Module,
    device: torch.device = None,
    dtype: torch.dtype = torch.complex64,
):
    factory_kwargs = {"device": device, "dtype": dtype}
    num_layers = 24
    num_heads = 16
    hidden_dim = 1024
    mlp_dim = 4096
    dropout = 0.0
    attention_dropout = 0.0
    norm_layer = c_nn.RMSNorm

    return c_nn.ViT(
        patch_embedder,
        num_layers,
        num_heads,
        hidden_dim,
        mlp_dim,
        dropout=dropout,
        attention_dropout=attention_dropout,
        norm_layer=norm_layer,
        **factory_kwargs
    )


def vit_h(
    patch_embedder: nn.Module,
    device: torch.device = None,
    dtype: torch.dtype = torch.complex64,
):
    factory_kwargs = {"device": device, "dtype": dtype}
    num_layers = 32
    num_heads = 16
    hidden_dim = 1280
    mlp_dim = 5120
    dropout = 0.0
    attention_dropout = 0.0
    norm_layer = c_nn.RMSNorm

    return c_nn.ViT(
        patch_embedder,
        num_layers,
        num_heads,
        hidden_dim,
        mlp_dim,
        dropout=dropout,
        attention_dropout=attention_dropout,
        norm_layer=norm_layer,
        **factory_kwargs
    )
