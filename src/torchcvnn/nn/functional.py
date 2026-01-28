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
from typing import Optional, Tuple
import math

# External imports
import torch
from torch import Tensor
import torch.nn.functional as F


def dropout(
    z: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False
) -> Tensor:
    if p < 0.0 or p > 1.0:
        raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
    mask = F.dropout(torch.ones(z.shape, device=z.device), p, training=training)
    return mask * z


def multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Optional[Tensor],
    in_proj_bias: Optional[Tensor],
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    training: bool = True,
    need_weights: bool = True,
    average_attn_weights: bool = True,
    is_causal: bool = False,
) -> Tuple[Tensor, Optional[Tensor]]:
    r"""Forward method for MultiHeadAttention.

    This function is adapted from pytorch torch.nn.functional.multi_head_attention_forward

    See :class:`torchcvnn.nn.MultiheadAttention` for details.

    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        need_weights: output attn_output_weights.
            Default: `True`
            Note: `needs_weight` defaults to `True`, but should be set to `False`
            For best performance when attention weights are not needed.
            *Setting needs_weights to `True`
            leads to a significant performance degradation.*
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across heads.
            Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an effect
            when ``need_weights=True.``. Default: True


    Shape:
        Inputs:
        - query: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, E)` or :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, E)` or :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.

        Outputs:
        - attn_output: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: Only returned when ``need_weights=True``. If ``average_attn_weights=True``, returns
          attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
          :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
          :math:`S` is the source sequence length. If ``average_attn_weights=False``, returns attention weights per
          head of shape :math:`(num_heads, L, S)` when input is unbatched or :math:`(N, num_heads, L, S)`.
    """

    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape

    assert (
        embed_dim == embed_dim_to_check
    ), f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"

    if isinstance(embed_dim, torch.Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = embed_dim.div(num_heads, rounding_mode="trunc")
    else:
        head_dim = embed_dim // num_heads
    assert (
        head_dim * num_heads == embed_dim
    ), f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"

    assert (
        key.shape == value.shape
    ), f"key shape {key.shape} does not match value shape {value.shape}"

    #
    # compute in-projection
    #
    assert (
        in_proj_weight is not None
    ), "use_separate_proj_weight is False but in_proj_weight is None"
    q, k, v = F._in_projection_packed(
        query, key, value, in_proj_weight, in_proj_bias
    )
    print("After in proj")
    print(f"Q shapes : {q.shape}") # tgt_len, B, embed_dim 
    print(f"K shapes : {k.shape}") # src_len, B, embed_dim
    print(f"V shapes : {v.shape}") # src_len, B, embed_dim

    #
    # reshape q, k, v for multihead attention and make them batch first
    #
    q = q.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)    # bsz * num_heads, tgt_len, embed_dim
    k = k.view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1) # bsz * num_heads, src_len, embed_dim
    v = v.view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1) # bsz * num_heads, src_len, embed_dim


    # update source sequence length after adjustments
    src_len = k.size(1)

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    #
    # (deep breath) calculate attention and out projection
    #

    # This is the "if need_weights" from the original pytorch code
    # We just adapt the case where the weights are needed
    # Indeed, since we are using specific implementations for computing
    # attention for the complex valued case, we cannot use the optimized versions
    # of the original pytorch code (flash attention or others)
    _B, _Nt, E = q.shape
    q_scaled = q * math.sqrt(1.0 / float(E))

    assert not (
        is_causal and attn_mask is None
    ), "FIXME: is_causal not implemented for need_weights"

    # For equation (8) from (Eilers et al. 2023),
    # We need to conjugate the keys before computing the dot product
    k = k.conj()

    attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
    print(f"Attn weights shape : {attn_output_weights.shape}")

    # And then take the real part of the result
    attn_output_weights = attn_output_weights.real

    attn_output_weights = F.softmax(attn_output_weights, dim=-1)
    if dropout_p > 0.0:
        attn_output_weights = dropout(attn_output_weights, p=dropout_p)

    # attn_output_weights are real valued while v are complex valued
    attn_output = torch.bmm(attn_output_weights.to(v.dtype), v)  # B, seq_len, embed_dim

    # torch.Size([231, 12, 16]) = [bsz x num_heads, tgt_len, head_dim]
    # Tgt_len  = 12, bsz = 11 , embed_dim = 336

    # print(attn_output.shape)
    # print(f"Tgt_len  = {tgt_len}, bsz = {bsz} , embed_dim = {embed_dim}")
    # sys.exit(-1)

    attn_output = (
        attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
    )
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
    attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1)) # seq_len , B, embed_dim

    # Early exist if we do not need the weights
    if not need_weights:
        return attn_output, None

    # Perform the extra computation only if the weights are needed

    # optionally average attention weights over heads
    attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
    if average_attn_weights:
        attn_output_weights = attn_output_weights.mean(dim=1)

    if not is_batched:
        # squeeze the output if input was unbatched
        attn_output = attn_output.squeeze(1)
        attn_output_weights = attn_output_weights.squeeze(0)
    return attn_output, attn_output_weights
