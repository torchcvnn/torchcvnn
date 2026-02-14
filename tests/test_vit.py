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

# Standard imports
from typing import Tuple, Union

# External imports
import torch
import torch.nn as nn

# Local imports
import torchcvnn.nn as c_nn
import torchcvnn.models as c_models


def test_vit_layer():

    batch_size = 16
    embed_dim = 768
    hidden_dim = 3072
    num_patches = 14 * 14

    X = torch.randn(num_patches, batch_size, embed_dim, dtype=torch.complex64)

    vit_layer = c_nn.ViTLayer(num_heads=8, embed_dim=embed_dim, hidden_dim=hidden_dim)
    out = vit_layer(X)

    assert out.shape == X.shape

class Transposer(nn.Module):
    def __init__(self, dim0, dim1):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)

def test_vit():
    batch_size = 16
    C, H, W = 3, 64, 64
    num_layers = 6
    patch_size = 16
    num_heads = 8
    embed_dim = 32
    hidden_dim = 4 * embed_dim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X = torch.randn(batch_size, C, H, W, dtype=torch.complex64, device=device)

    patch_embedding = nn.Sequential(
        nn.Conv2d(
            C,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            dtype=torch.complex64,
        ),
        c_nn.LayerNorm(embed_dim),
        nn.Flatten(2),
        Transposer(1, 2)
    )
    patch_embedding = patch_embedding.to(device) # B, C, embed_dim, H//patch_size, W//patch_size

    vit = c_nn.ViT(
        patch_embedding,
        num_layers=num_layers,
        num_heads=num_heads,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
    )
    vit = vit.to(device)
    out = vit(X)

    num_patches = (H // patch_size) * (W // patch_size)
    assert out.shape == (batch_size, num_patches, embed_dim)


class PatchEmbedder(nn.Module):

    def __init__(
        self,
        image_size: Union[int, Tuple[int, int]],
        cin,
        embed_dim,
        patch_size,
        norm_layer: nn.Module = c_nn.LayerNorm,
        device: torch.device = None,
        dtype=torch.complex64,
    ):
        super(PatchEmbedder, self).__init__()

        factory_kwargs = {"device": device, "dtype": dtype}

        if isinstance(image_size, int):
            image_size = (image_size, image_size)

        self.embedder = nn.Sequential(
            norm_layer([cin, *image_size], **factory_kwargs),
            nn.Conv2d(
                cin,
                embed_dim,
                kernel_size=patch_size,
                stride=patch_size,
                **factory_kwargs,
            ),
            # norm_layer([embed_dim], **factory_kwargs),
            norm_layer(
                [embed_dim, image_size[0] // patch_size, image_size[1] // patch_size],
                **factory_kwargs,
            ),
            nn.Flatten(2),
            Transposer(1, 2)
        )

    def forward(self, x):
        return self.embedder(x)


def test_vit_bhl():
    batch_size = 2
    C, H, W = 3, 64, 64

    device = torch.device("cpu")

    X = torch.randn(batch_size, C, H, W, dtype=torch.complex64, device=device)

    vit_models = [("vit_b", 768)]  # , ("vit_l", 1024), ("vit_h", 1280)]

    for mvit, embed_dim in vit_models:
        for patch_size in [16, 32]:
            patch_embedding = PatchEmbedder(
                (H, W), C, embed_dim, patch_size, c_nn.LayerNorm, device=device
            )

            vit = eval(f"c_models.{mvit}(patch_embedding).to(device)")

            out = vit(X)

            num_patches = (H // patch_size) * (W // patch_size)
            assert out.shape == (batch_size, num_patches, embed_dim)


if __name__ == "__main__":
    test_vit_layer()
    test_vit()
    test_vit_bhl()
