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

# External imports
import torch
from torch.nn.modules.transformer import TransformerDecoder, TransformerEncoder

# Local imports
import torchcvnn.nn as c_nn


def test_multihead_scaleddotproduct_selfattention():
    nhead = 8
    seq_len = 10
    batch_size = 32
    num_features = 512

    multihead_attn = c_nn.MultiheadAttention(embed_dim=num_features, num_heads=nhead)
    src = torch.rand(seq_len, batch_size, num_features, dtype=torch.complex64)
    attn_output, attn_output_weights = multihead_attn(src, src, src)

    assert attn_output.shape == (seq_len, batch_size, num_features)


# def test_multihead_masks():
#    nheads = 8
#
#    src_seq_len = 10
#    tgt_seq_len = 20
#
#    embed_dim = 16  # multiple of nheads, so that head_dim = embed_dim // nheads
#    kdim = 12
#    vdim = 13
#
#    batch_size = 32
#
#    # Batch_first = False
#    query = torch.rand(tgt_seq_len, batch_size, embed_dim, dtype=torch.complex64)
#    key = torch.rand(src_seq_len, batch_size, kdim, dtype=torch.complex64)
#    value = torch.rand(src_seq_len, batch_size, vdim, dtype=torch.complex64)
#
#    key_padding_mask = torch.rand(batch_size, src_seq_len) > 0.5
#
#    multihead_attn = c_nn.MultiheadAttention(
#        embed_dim=embed_dim, num_heads=nheads, kdim=kdim, vdim=vdim
#    )
#    attn_output, attn_output_weights = multihead_attn(
#        query, key, value, key_padding_mask=key_padding_mask
#    )
#
#    assert attn_output.shape == (tgt_seq_len, batch_size, embed_dim)


def test_multihead_scaleddotproduct():
    nheads = 8

    src_seq_len = 10
    tgt_seq_len = 20

    embed_dim = 16  # multiple of nheads, so that head_dim = embed_dim // nheads
    kdim = 12
    vdim = 13

    batch_size = 32

    # Batch_first = False
    query = torch.rand(tgt_seq_len, batch_size, embed_dim, dtype=torch.complex64)
    key = torch.rand(src_seq_len, batch_size, kdim, dtype=torch.complex64)
    value = torch.rand(src_seq_len, batch_size, vdim, dtype=torch.complex64)

    multihead_attn = c_nn.MultiheadAttention(
        embed_dim=embed_dim, num_heads=nheads, kdim=kdim, vdim=vdim
    )
    attn_output, attn_output_weights = multihead_attn(query, key, value)

    assert attn_output.shape == (tgt_seq_len, batch_size, embed_dim)

    # Batch_first = True
    query = torch.rand(batch_size, tgt_seq_len, embed_dim, dtype=torch.complex64)
    key = torch.rand(batch_size, src_seq_len, kdim, dtype=torch.complex64)
    value = torch.rand(batch_size, src_seq_len, vdim, dtype=torch.complex64)

    multihead_attn = c_nn.MultiheadAttention(
        embed_dim=embed_dim, num_heads=nheads, kdim=kdim, vdim=vdim, batch_first=True
    )

    attn_output, attn_output_weights = multihead_attn(query, key, value)

    assert attn_output.shape == (batch_size, tgt_seq_len, embed_dim)


def test_transformer_encoder_layer():
    nhead = 8
    seq_len = 10
    batch_size = 32
    num_features = 512

    encoder_layer = c_nn.TransformerEncoderLayer(d_model=num_features, nhead=nhead)
    src = torch.rand(seq_len, batch_size, num_features, dtype=torch.complex64)
    out = encoder_layer(src)

    assert out.shape == (seq_len, batch_size, num_features)

    encoder_layer = c_nn.TransformerEncoderLayer(
        d_model=num_features, nhead=nhead, batch_first=True
    )
    src = torch.rand(batch_size, seq_len, num_features, dtype=torch.complex64)
    out = encoder_layer(src)

    assert out.shape == (batch_size, seq_len, num_features)


def test_transformer_encoder():
    nhead = 8
    seq_len = 10
    batch_size = 32
    num_features = 512

    encoder_layer = c_nn.TransformerEncoderLayer(d_model=num_features, nhead=nhead)
    transformer_encoder = TransformerEncoder(encoder_layer, num_layers=4)

    src = torch.randn((seq_len, batch_size, num_features), dtype=torch.complex64)
    out = transformer_encoder(src)
    assert out.shape == (seq_len, batch_size, num_features)


def test_transformer_decoder_layer():
    nhead = 8
    src_seq_len = 10
    tgt_seq_len = 20
    batch_size = 32
    num_features = 512

    decoder_layer = c_nn.TransformerDecoderLayer(d_model=num_features, nhead=nhead)
    memory = torch.rand(src_seq_len, batch_size, num_features, dtype=torch.complex64)
    tgt = torch.rand(tgt_seq_len, batch_size, num_features, dtype=torch.complex64)
    out = decoder_layer(tgt, memory)

    assert out.shape == (tgt_seq_len, batch_size, num_features)

    decoder_layer = c_nn.TransformerDecoderLayer(
        d_model=num_features, nhead=nhead, batch_first=True
    )
    memory = torch.rand(batch_size, src_seq_len, num_features, dtype=torch.complex64)
    tgt = torch.rand(batch_size, tgt_seq_len, num_features, dtype=torch.complex64)
    out = decoder_layer(tgt, memory)

    assert out.shape == (batch_size, tgt_seq_len, num_features)


def test_transformer_decoder():
    nhead = 8
    src_seq_len = 10
    tgt_seq_len = 20
    batch_size = 32
    num_features = 512

    decoder_layer = c_nn.TransformerDecoderLayer(d_model=num_features, nhead=nhead)
    transformer_decoder = TransformerDecoder(decoder_layer, num_layers=4)
    memory = torch.rand((src_seq_len, batch_size, num_features), dtype=torch.complex64)
    tgt = torch.rand((tgt_seq_len, batch_size, num_features), dtype=torch.complex64)
    out = transformer_decoder(tgt, memory)

    assert out.shape == (tgt_seq_len, batch_size, num_features)


def test_transformer():
    nhead = 8
    src_seq_len = 10
    tgt_seq_len = 20
    batch_size = 32
    num_features = 512

    transformer = c_nn.Transformer(d_model=num_features, nhead=nhead, batch_first=False)

    src = torch.rand((src_seq_len, batch_size, num_features), dtype=torch.complex64)
    tgt = torch.rand((tgt_seq_len, batch_size, num_features), dtype=torch.complex64)
    out = transformer(src, tgt)

    assert out.shape == (tgt_seq_len, batch_size, num_features)

    transformer = c_nn.Transformer(d_model=num_features, nhead=nhead, batch_first=True)

    src = torch.rand((batch_size, src_seq_len, num_features), dtype=torch.complex64)
    tgt = torch.rand((batch_size, tgt_seq_len, num_features), dtype=torch.complex64)
    out = transformer(src, tgt)

    assert out.shape == (batch_size, tgt_seq_len, num_features)


if __name__ == "__main__":
    test_multihead_scaleddotproduct_selfattention()
    test_multihead_scaleddotproduct()
    # test_multihead_masks()
    test_transformer_encoder_layer()
    test_transformer_encoder()
    test_transformer_decoder_layer()
    test_transformer_decoder()
    test_transformer()
