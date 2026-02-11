

import torch.nn as nn
import torch
from torch import Tensor
import torchcvnn.nn as c_nn
from typing import Tuple, Union, List
import torchinfo


import embedders

class Attention(nn.Module):
    """Complex-valued attention layer for Vision Transformer, as proposed in "Building Blocks for a Complex-Valued Transformer Architecture" by Eilers et al.

    Args:
        embed_dim (int): Embedding dimension
        num_heads (int): Number of attention heads
    """
    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.q_norm = c_nn.RMSNorm(self.head_dim)
        self.k_norm = c_nn.RMSNorm(self.head_dim)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, dtype=torch.complex64)

    def forward(self, x: Tensor) -> Tensor:
        B, N, _ = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4) # (3, B, num_heads, num_patches, head_dim)
            .contiguous()
        )
        q, k, v = qkv.unbind(0) # (B, num_heads, num_patches, head_dim)
        q, k = self.q_norm(q), self.k_norm(k)
        return self.scaled_dot_product_attention(q, k, v)

    def scaled_dot_product_attention(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        out = ((q @ k.transpose(-2, -1).conj()).real * self.scale).softmax(dim=-1)
        weighted_v =  (out.to(torch.complex64) @ v) # B, num_heads, N, head_dim
        out = torch.flatten(weighted_v.transpose(1, 2), start_dim=-2, end_dim=-1) # B, N, embed_dim

        return out


class Block(nn.Module):
    """Vision Transformer block.

    Args:
        embed_dim (int): Embedded dimension
        hidden_dim (int): Hidden dimension
        num_heads (int): Number of attention heads
        dropout (float): Dropout rate
    """
    def __init__(
        self, embed_dim: int, hidden_dim: int, num_heads: int, dropout: float = 0.0
    ) -> None:
        super().__init__()

        self.attn = Attention(embed_dim, num_heads)
        self.layer_norm = c_nn.RMSNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim, dtype=torch.complex64),
            c_nn.CGELU(),
            c_nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim, dtype=torch.complex64),
            c_nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        attn = self.attn(x)
        x = x + attn
        x = x + self.linear(self.layer_norm(x))
        # inp_x = self.layer_norm(x)
        # x = x + self.attn(inp_x, inp_x, inp_x)[0]
        # x = x + self.linear(self.layer_norm(x))

        return x


class VisionTransformer(nn.Module):
    """Vision Transformer model implementation based on the paper "An image is worth 16x16 words: Transformers for image recognition at scale" by Dosovitskiy et al.
    It is adapted to work with complex-valued inputs, with complex-valued blocks from torchcvnn, and a complex-valued attention layer.

    Args:
        opt (ArgumentParser): model configuration defined in the parser.
        num_classes (int): Number of classes in the dataset.
    """
    # This module was implemented based on 
    def __init__(self, opt: dict, num_classes: int) -> None:

        super().__init__()

        patch_size = opt["patch_size"]
        input_size = opt["input_size"]
        embed_dim = opt["embed_dim"]
        hidden_dim = opt["hidden_dim"]
        num_layers = opt["num_layers"]
        num_heads = opt["num_heads"]
        num_channels = opt["num_channels"]
        dropout = opt["dropout"]
        # attention_dropout = opt["attention_dropout"]
        # norm_layer = opt["norm_layer"]

        self.patch_size = patch_size
        assert (
            input_size % patch_size == 0
        ), "Image size must be divisible by the patch size"
        self.num_patches = (input_size // patch_size) ** 2
        # Define whether to use traditional ViT or hybrid-ViT
        self.patch_embedder = embedders.Image2Patch(patch_size)
        input_layer_channels = num_channels * (patch_size**2)
        # Input layer
        self.input_layer = nn.Linear(
            input_layer_channels, embed_dim, dtype=torch.complex64
        )
        # Tranformer blocks
        self.transformer = nn.Sequential(
            *(
                Block(
                    embed_dim, hidden_dim, num_heads, dropout=dropout
                )
                for _ in range(num_layers)
            )
        )
        # MLP head
        self.mlp_head = nn.Sequential(
            c_nn.RMSNorm(embed_dim),
            nn.Linear(embed_dim, num_classes, dtype=torch.complex64),
        )
        self.dropout = c_nn.Dropout(dropout)
        # Class tokens
        self.cls_token = nn.Parameter(
            torch.rand(1, 1, embed_dim, dtype=torch.complex64)
        )
        # Positional embeddings
        self.pos_embedding = nn.Parameter(
            torch.rand(1, 1 + self.num_patches, embed_dim, dtype=torch.complex64)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.patch_embedder(x)
        B, T, _ = x.shape
        x = self.input_layer(x)

        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embedding

        x = self.dropout(x)
        x = self.transformer(x)

        cls = x[:, 0] # position of cls_token
        return self.mlp_head(cls)



if __name__ == "__main__":
    opt = {
        "patch_size": 7,
        "input_size": 28,
        "embed_dim": 16,
        "hidden_dim": 32,
        "num_layers": 3,
        "num_heads": 8,
        "num_channels": 1,
        "dropout": 0.3,
        # "attention_dropout": 0.1,
        # "norm_layer": "rms_norm",
        # "model_type": "vit"
    }
    num_classes = 10

    model = VisionTransformer(opt, num_classes)
    torchinfo.summary(model, depth=3)
    B, C, H, W = 12, 1, 28, 28
    X = torch.zeros((B, C, H, W), dtype=torch.complex64)

    y = model(X)
    print(y.shape)
    print(y.dtype)
