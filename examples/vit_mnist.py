# MIT License

# Copyright (c) 2023 Jérémy Fix

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

"""
# Example using a complex valued Vision transformer to classify MNIST. 



Requires dependencies :
    python3 -m pip install torchvision tqdm
"""

# Standard imports
import random
import sys
from typing import List, Tuple, Union

# External imports
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.v2 as v2_transforms

import torchcvnn.nn as c_nn
import torchcvnn.models as c_models

# Local imports
import utils


class PseudoNorm(nn.Module):

    def __init__(self, dim, dtype=None, device=None):
        super(PseudoNorm, self).__init__()

    def forward(self, x):
        return x


class PatchEmbedder(nn.Module):

    def __init__(
        self,
        image_size: Union[int, Tuple[int, int]],
        cin,
        hidden_dim,
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
                hidden_dim,
                kernel_size=patch_size,
                stride=patch_size,
                **factory_kwargs,
            ),
            norm_layer(
                [hidden_dim, image_size[0] // patch_size, image_size[1] // patch_size],
                **factory_kwargs,
            ),
        )

    def rope_embedding(self, H, W, hidden_dim, device):
        """
        Computes and return the 2D rotary positional embedding RoPE from

        "Rotary position embedding for Vision Transformer", Heo et al 2024, ECCV

        Args:
            H (int): Height of the image
            W (int): Width of the image
            hidden_dim (int): Hidden dimension of the model

        Returns:
            torch.Tensor: Positional embeddings for the patches
        """
        # Frequency scale is 10000 in original "Attention is all you need paper"
        # but downscaled to 100 in RoPE paper
        frequency_scale = 100

        pos_H = torch.arange(H, dtype=torch.float32, device=device)
        pos_W = torch.arange(W, dtype=torch.float32, device=device)

        # Compute the positional encoding
        theta_t = frequency_scale ** (
            torch.arange(0, hidden_dim, 2, dtype=torch.float32, device=device)
            / (hidden_dim / 2)
        )

        # Apply the sine/cosine embedding
        emb = torch.exp(1j * theta_t)

        # Compute the positional embedding
        emb_H = emb[:, None] ** pos_H[None, :]  # (hidden_dim//2, H)
        emb_W = emb[:, None] ** pos_W[None, :]  # (hidden_dim//2, W)

        # The even dimensions of the features use the encoding of the height
        # while the odd dimensions use the encoding of the width
        # So that we interleave the emb_H and emb_W
        embeddings = torch.zeros(hidden_dim, H, W, dtype=torch.complex64, device=device)
        embeddings[0::2, :, :] = emb_H[:, :, None]
        embeddings[1::2, :, :] = emb_W[:, None, :]

        return embeddings

    def forward(self, x):
        patch_embeddings = self.embedder(x)  # (B, embed_dim, num_patch_H, num_patch_W)

        num_patches_H, num_patches_W = patch_embeddings.shape[2:]

        # Adds the positionnal embedding
        pos_emb = self.rope_embedding(
            num_patches_H, num_patches_W, patch_embeddings.shape[1], device=x.device
        )
        patch_embeddings = patch_embeddings + pos_emb

        return patch_embeddings


class Model(nn.Module):

    def __init__(self):
        super().__init__()

        # The hidden_dim must be adapted to the hidden_dim of the ViT model
        # It is used as the output dimension of the patch embedder but must match
        # the expected hidden dim of your ViT
        hidden_dim = 32
        dropout = 0.1
        attention_dropout = 0.1
        # norm_layer = PseudoNorm
        norm_layer = c_nn.RMSNorm
        # norm_layer = c_nn.LayerNorm
        patch_size = 7

        embedder = PatchEmbedder(28, 1, hidden_dim, patch_size, norm_layer=norm_layer)

        # For using an off-the shelf ViT model, you can use the following code
        # If you go this way, do not forget to adapt the hidden_dim above
        # For vit_t, it is 192
        # self.backbone = c_models.vit_t(embedder)

        # For a custom ViT model, you can use the following code
        # If you go this way, do not forget to adapt the hidden_dim above
        # You can reduce it to 32 for example

        num_layers = 3
        num_heads = 8
        mlp_dim = 4 * hidden_dim

        self.backbone = c_nn.ViT(
            embedder,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            norm_layer=norm_layer,
        )

        # A Linear decoding head to project on the logits
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 10, dtype=torch.complex64), c_nn.Mod()
        )

    def forward(self, x):
        features = self.backbone(x)  # B, num_patches, hidden_dim

        # Global average pooling of the patches encoding
        mean_features = features.mean(dim=1)

        return self.head(mean_features)


def train():
    """
    Train function

    Sample output :
        ```.bash
        (venv) me@host:~$ python mnist.py
        Logging to ./logs/CMNIST_0
        >> Training
        100%|████| 844/844 [00:15<00:00, 53.13it/s]
        >> Testing
        [Step 0] Train : CE  1.21 Acc  0.64 | Valid : CE  0.57 Acc  0.84 | Test : CE  0.56 Acc  0.84[>> BETTER <<]
        >> Training
        100%|████| 844/844 [00:15<00:00, 53.64it/s]
        >> Testing
        [Step 1] Train : CE  0.39 Acc  0.89 | Valid : CE  0.31 Acc  0.91 | Test : CE  0.29 Acc  0.92[>> BETTER <<]
        >> Training
        100%|████| 844/844 [00:15<00:00, 54.35it/s]
        >> Testing
        [Step 2] Train : CE  0.24 Acc  0.93 | Valid : CE  0.20 Acc  0.94 | Test : CE  0.21 Acc  0.94[>> BETTER <<]
        [...]
        ```

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    valid_ratio = 0.1
    batch_size = 6
    epochs = 10
    cdtype = torch.complex64

    # Dataloading
    train_valid_dataset = torchvision.datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=v2_transforms.Compose(
            [v2_transforms.PILToTensor(), v2_transforms.ToDtype(cdtype)]
        ),
    )
    test_dataset = torchvision.datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=v2_transforms.Compose(
            [v2_transforms.PILToTensor(), v2_transforms.ToDtype(cdtype)]
        ),
    )

    all_indices = list(range(len(train_valid_dataset)))
    random.shuffle(all_indices)
    split_idx = int(valid_ratio * len(train_valid_dataset))
    valid_indices, train_indices = all_indices[:split_idx], all_indices[split_idx:]

    # Train dataloader
    train_dataset = torch.utils.data.Subset(train_valid_dataset, train_indices)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    # Valid dataloader
    valid_dataset = torch.utils.data.Subset(train_valid_dataset, valid_indices)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False
    )

    # Test dataloader
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    # Model
    model = Model().to(device)

    # Loss, optimizer, callbacks
    f_loss = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=3e-4)
    logpath = utils.generate_unique_logpath("./logs", "CMNIST")
    print(f"Logging to {logpath}")
    checkpoint = utils.ModelCheckpoint(model, logpath, 4, min_is_best=True)

    # Training loop
    for e in range(epochs):
        print(">> Training")
        train_loss, train_acc = utils.train_epoch(
            model, train_loader, f_loss, optim, device
        )

        print(">> Testing")
        valid_loss, valid_acc = utils.test_epoch(model, valid_loader, f_loss, device)
        test_loss, test_acc = utils.test_epoch(model, test_loader, f_loss, device)
        updated = checkpoint.update(valid_loss)
        better_str = "[>> BETTER <<]" if updated else ""

        print(
            f"[Step {e}] Train : CE {train_loss:5.2f} Acc {train_acc:5.2f} | Valid : CE {valid_loss:5.2f} Acc {valid_acc:5.2f} | Test : CE {test_loss:5.2f} Acc {test_acc:5.2f}"
            + better_str
        )


if __name__ == "__main__":
    train()
