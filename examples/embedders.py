import torch.nn as nn

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
