

import torch.nn as nn
import torch
from torch import Tensor
import torchcvnn.nn as c_nn
from typing import Tuple, Union, List
from argparse import ArgumentParser


class Image2Patch(nn.Module):
    """Converts an image into patches.

    Args:
        patch_size (int): size of the patch
        flatten_channels (bool): whether to flatten the channels of the patch representation
    """
    def __init__(self, patch_size: int, flatten_channels: bool = True):
        super().__init__()
        self.patch_size = patch_size
        self.flatten_channels = flatten_channels

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        assert (
            H // self.patch_size != 0 and W // self.patch_size != 0
        ), f"Image height and width are {H, W}, which is not a multiple of the patch size"
        # Shape of x: (B, C, H, W)
        # Reshape to (B, C, number of patch along H, patch_size, number of patch along W, patch_size)
        x = x.reshape(B, C, H // self.patch_size, self.patch_size, W // self.patch_size, self.patch_size)
        # Permute axis. Shape of x after permute: (B, number of patch along H, number of patch along W, C, patch_size, patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5)
        # Flatten 1st and 2nd axis to obtain to total amount of patches. Shape of x after flatten: (B, number of patch, C, patch_size, patch_size)
        x = x.flatten(1, 2)

        if self.flatten_channels:
            # Flatten to obtain a 1D patch representation. Shape of x after flatten: (B, number of patch, C * patch_size * patch_size)
            return x.flatten(2, 4)
        else:
            # Return full patch representation. Shape of x: (B, number of patch, C, patch_size, patch_size)
            return x


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
        return out.to(torch.complex64) @ v


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
        attn = self.attn(x).transpose(1, 2).reshape(B, N, C)
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
        hidden_dim = opt["hidden_dim"]
        num_layers = opt["num_layers"]
        num_heads = opt["num_heads"]
        num_channels = opt["num_channels"]
        dropout = opt["dropout"]
        # attention_dropout = opt["attention_dropout"]
        # norm_layer = opt["norm_layer"]
        model_type = opt["model_type"]

        self.patch_size = patch_size
        assert (
            input_size % patch_size == 0
        ), "Image size must be divisible by the patch size"
        self.num_patches = (input_size // patch_size) ** 2
        # Define whether to use traditional ViT or hybrid-ViT
        if "hybrid" in model_type:
            self.patch_embedder = ConvStem(num_channels, hidden_dim, patch_size)
            self.embed_dim = int(num_channels * (patch_size**2) / 2)
            input_layer_channels = hidden_dim
        else:
            self.patch_embedder = Image2Patch(patch_size)
            self.embed_dim = int(hidden_dim / 2)
            input_layer_channels = num_channels * (patch_size**2)
        # Input layer
        self.input_layer = nn.Linear(
            input_layer_channels, self.embed_dim, dtype=torch.complex64
        )
        # Tranformer blocks
        self.transformer = nn.Sequential(
            *(
                Block(
                    self.embed_dim, hidden_dim, num_heads, dropout=dropout
                )
                for _ in range(num_layers)
            )
        )
        # MLP head
        self.mlp_head = nn.Sequential(
            c_nn.RMSNorm(self.embed_dim),
            nn.Linear(self.embed_dim, num_classes, dtype=torch.complex64),
        )
        self.dropout = c_nn.Dropout(dropout)
        # Class tokens
        self.cls_token = nn.Parameter(
            torch.rand(1, 1, self.embed_dim, dtype=torch.complex64)
        )
        # Positional embeddings
        self.pos_embedding = nn.Parameter(
            torch.rand(1, 1 + self.num_patches, self.embed_dim, dtype=torch.complex64)
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


class ConvStem(nn.Module):
    """
        Convolutional Stem to replace Image2Patch.
        This converts vanilla Vision Transformers into a hybrid model.
        Stem layers work as a compression mechanism over the initial image, they typically compute convolution with large kernel size and/or stride. 
        This leads to a better spatial dimension, which could be help the Vision Transformer to generalize better.

        Args:
            in_channels (int): Number of input channels. For MSTAR dataset, it is 1.
            hidden_dim (int): Dimension of the hidden dimension of the ViT.
            patch_size (int): Patch size used to split the image.
        """
    def __init__(self, in_channels, hidden_dim, patch_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_dim // 2,
                kernel_size=7,
                stride=2,
                padding=3,
                dtype=torch.complex64,
            ),
            c_nn.BatchNorm2d(hidden_dim // 2, track_running_stats=False),
            c_nn.modReLU(),
            nn.Conv2d(
                hidden_dim // 2,
                hidden_dim,
                kernel_size=3,
                stride=patch_size // 2,
                padding=1,
                dtype=torch.complex64,
            ),
            c_nn.BatchNorm2d(hidden_dim, track_running_stats=False),
            c_nn.modReLU(),
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Patch embeddings of shape (B, hidden_dim, H, W).
        """
        # Apply the convolutional stem. Output shape: (B, hidden_dim, num_patches_H, num_patches_W)
        x = self.conv(x)
        # Flatten the pathces. Output shape: (B, hidden_dim, num_patches_H * num_patches_W)
        x = x.flatten(2)
        # Rearrange to (B, num_patches_H * num_patches_W, hidden_dim)
        x = x.transpose(1, 2)
        return x

if __name__ == "__main__":
    opt = {
        "patch_size": 16,
        "input_size": 128,
        "hidden_dim": 32,
        "num_layers": 3,
        "num_heads": 8,
        "num_channels": 1,
        "dropout": 0.3,
        # "attention_dropout": 0.1,
        # "norm_layer": "rms_norm",
        "model_type": "hybrid-vit"
    }
    num_classes = 10

    model = VisionTransformer(opt, num_classes)
    B, C, H, W = 10, 1, 128, 128
    X = torch.zeros((B, C, H, W), dtype=torch.complex64)

    model(X)



    
