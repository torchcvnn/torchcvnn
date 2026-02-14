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
from typing import Callable

# External imports
import torch
import torch.nn as nn

# Local imports
from .normalization import LayerNorm
from .activation import MultiheadAttention, CGELU
from .dropout import Dropout


class ViTLayer(nn.Module):

    def __init__(
        self,
        num_heads: int,
        embed_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        norm_layer: Callable[..., nn.Module] = LayerNorm,
        activation_fn: Callable[[], nn.Module] = CGELU,
        device: torch.device = None,
        dtype: torch.dtype = torch.complex64,
    ) -> None:
        """
        The ViT layer cascades a multi-head attention block with a feed-forward network.

        Args:
            num_heads: Number of heads in the multi-head attention block.
            embed_dim: Hidden dimension of the transformer.
            hidden_dim: Hidden dimension of the feed-forward network.
            dropout: Dropout rate (default: 0.0).
            attention_dropout: Dropout rate in the attention block (default: 0.0).
            norm_layer: Normalization layer (default :py:class:`LayerNorm`).

        .. math::

            x & = x + \\text{attn}(\\text{norm1}(x))\\\\
            x & = x + \\text{ffn}(\\text{norm2}(x))

        The FFN block is a two-layer MLP with a modReLU activation function.

        """
        super(ViTLayer, self).__init__()

        factory_kwargs = {"device": device, "dtype": dtype}

        self.norm1 = norm_layer(embed_dim)
        self.attn = MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            norm_layer=norm_layer,
            batch_first=True,
            **factory_kwargs
        )
        self.dropout1 = Dropout(dropout)

        self.ffn = nn.Sequential(
            norm_layer(embed_dim),
            nn.Linear(embed_dim, hidden_dim, **factory_kwargs),
            activation_fn(),
            Dropout(dropout),
            norm_layer(hidden_dim),
            nn.Linear(hidden_dim, embed_dim, **factory_kwargs),
            activation_fn(),
            Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass through the layer using pre-normalization.

        Args:
            x: Input tensor of shape (B, seq_len, hidden_dim)
        """
        norm_x = self.norm1(x)
        x = x + self.dropout1(self.attn(norm_x, norm_x, norm_x, need_weights=False))
        x = x + self.ffn(x)

        return x


class ViT(nn.Module):

    def __init__(
        self,
        patch_embedder: nn.Module,
        num_layers: int,
        num_heads: int,
        embed_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        norm_layer: Callable[..., nn.Module] = LayerNorm,
        device: torch.device = None,
        dtype: torch.dtype = torch.complex64,
    ):
        """
        Vision Transformer model. This implementation does not contain any head.

        For classification, you can for example compute a global average of the output embeddings :

        .. code-block:: python

            backbone = c_nn.ViT(
                embedder,
                num_layers,
                num_heads,
                embed_dim,
                hidden_dim,
                dropout=dropout,
                attention_dropout=attention_dropout,
                norm_layer=norm_layer,
            )

            # A Linear decoding head to project on the logits
            head = nn.Sequential(
                nn.Linear(embed_dim, 10, dtype=torch.complex64), c_nn.Mod()
            )

            x = torch.randn(B C, H, W)
            features = backbone(x)  # B, num_patches, embed_dim

            # Global average pooling of the patches encoding
            mean_features = features.mean(dim=1) # B, embed_dim

            head(mean_features)

        Args:
            patch_embedder: PatchEmbedder instance.
            num_layers: Number of layers in the transformer.
            num_heads: Number of heads in the multi-head attention block.
            embed_dim: Hidden dimension of the transformer.
            hidden_dim: Hidden dimension of the feed-forward network.
            dropout: Dropout rate (default: 0.0).
            attention_dropout: Dropout rate in the attention block (default: 0.0).
            norm_layer: Normalization layer (default :py:class:`LayerNorm`).
        """
        super(ViT, self).__init__()

        factory_kwargs = {"device": device, "dtype": dtype}

        self.patch_embedder = patch_embedder
        self.dropout = Dropout(dropout)
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(
                ViTLayer(
                    num_heads=num_heads,
                    embed_dim=embed_dim,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    norm_layer=norm_layer,
                    **factory_kwargs
                )
            )
        self.layers = nn.Sequential(*self.layers)
        self.norm = norm_layer(embed_dim, **factory_kwargs)

    def forward(self, x):
        # x : (B, C, H, W)
        embedding = self.patch_embedder(x)  # (B, src_len, embed_dim)

        out = self.layers(self.dropout(embedding))

        out = self.norm(out)

        return out
