import torch
import torch.nn as nn
import torchcvnn.nn as c_nn
import torchinfo

import embedders

class PatchEmbedderPos(nn.Module):

    def __init__(self, num_patches, patch_size, hidden_dim, num_channels, dropout):
        super().__init__()
        
        self.i2p = embedders.Image2Patch(patch_size) # (B, number of patch, C * patch_size * patch_size)

        input_layer_channels = num_channels * (patch_size**2)
        embed_dim = hidden_dim # int(hidden_dim / 2)
        self.input_layer = nn.Linear(
            input_layer_channels, embed_dim, dtype=torch.complex64
        )

        self.dropout = c_nn.Dropout(dropout)
        # Class tokens
        self.cls_token = nn.Parameter(
            torch.rand(1, 1, embed_dim, dtype=torch.complex64)
        )
        # Positional embeddings
        self.pos_embedding = nn.Parameter(
            torch.rand(1, 1 + num_patches, embed_dim, dtype=torch.complex64)
        )

    def forward(self, x):
        x = self.i2p(x) # B, number of patch, C * patch_size * patch_size
        x = self.input_layer(x) # B, number_of_patch, embed_dim

        B = x.shape[0]
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1) # B, 1 + number_of_patch, embed_dim
        x = x + self.pos_embedding

        x = self.dropout(x) # B, 1 + number_of_patch, embed_dim

        return x

class Model(nn.Module):

    def __init__(self, opt: dict, num_classes: int):
        super().__init__()

        patch_size = opt["patch_size"]
        input_size = opt["input_size"]
        embed_dim = opt["embed_dim"]
        hidden_dim = opt["hidden_dim"]
        num_layers = opt["num_layers"]
        num_heads = opt["num_heads"]
        num_channels = opt["num_channels"]
        dropout = opt["dropout"]
        attention_dropout = opt["attention_dropout"]
        # norm_layer = opt["norm_layer"]

        # The hidden_dim must be adapted to the hidden_dim of the ViT model
        # It is used as the output dimension of the patch embedder but must match
        # the expected hidden dim of your ViT
        # norm_layer = PseudoNorm
        norm_layer = c_nn.RMSNorm
        # norm_layer = c_nn.LayerNorm

        num_patches = (input_size // patch_size) ** 2
        embedder = PatchEmbedderPos(num_patches, patch_size, embed_dim, num_channels, dropout)
        # embedder = embedders.PatchEmbedder(input_size, 
        #                                    1, 
        #                                    embed_dim, 
        #                                    patch_size, 
        #                                    norm_layer=norm_layer)

        # For using an off-the shelf ViT model, you can use the following code
        # If you go this way, do not forget to adapt the embed_dim above
        # For vit_t, it is 192
        # self.backbone = c_models.vit_t(embedder)

        # For a custom ViT model, you can use the following code
        # If you go this way, do not forget to adapt the embed_dim above
        # You can reduce it to 32 for example

        self.backbone = c_nn.ViT(
            embedder,
            num_layers,
            num_heads,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            norm_layer=norm_layer,
        )

        # A Linear decoding head to project on the logits
        self.head = nn.Sequential(
            nn.Linear(embed_dim, num_classes, dtype=torch.complex64)
        )

    def forward(self, x):
        features = self.backbone(x)  # B, num_patches, embed_dim

        # print(features.shape)
    
        # Global average pooling of the patches encoding
        # mean_features = features.mean(dim=1)
        # return self.head(mean_features)

        cls_features = features[:, 0]
        return self.head(cls_features)

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
        "attention_dropout": 0.1,
        "norm_layer": "rms_norm",
    }
    num_classes = 10

    model = Model(opt, num_classes)
    torchinfo.summary(model, depth=4)
    B, C, H, W = 12, 1, 28, 28
    X = torch.zeros((B, C, H, W), dtype=torch.complex64)

    y = model(X)
    print(y.shape)
    print(y.dtype)

