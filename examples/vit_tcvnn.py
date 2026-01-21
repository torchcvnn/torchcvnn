import torch
import torch.nn as nn
import torchcvnn.nn as c_nn

import embedders

class Model(nn.Module):

    def __init__(self, opt: dict, num_classes: int):
        super().__init__()

        patch_size = opt["patch_size"]
        input_size = opt["input_size"]
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

        embedder = embedders.PatchEmbedder(input_size, 
                                           1, 
                                           hidden_dim, 
                                           patch_size, 
                                           norm_layer=norm_layer)

        # For using an off-the shelf ViT model, you can use the following code
        # If you go this way, do not forget to adapt the hidden_dim above
        # For vit_t, it is 192
        # self.backbone = c_models.vit_t(embedder)

        # For a custom ViT model, you can use the following code
        # If you go this way, do not forget to adapt the hidden_dim above
        # You can reduce it to 32 for example

        mlp_dim = 4*hidden_dim

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
            nn.Linear(hidden_dim, num_classes, dtype=torch.complex64)
        )

    def forward(self, x):
        features = self.backbone(x)  # B, num_patches, hidden_dim

        # Global average pooling of the patches encoding
        mean_features = features.mean(dim=1)

        return self.head(mean_features)

if __name__ == "__main__":
    opt = {
        "patch_size": 7,
        "input_size": 28,
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
    B, C, H, W = 10, 1, 28, 28
    X = torch.zeros((B, C, H, W), dtype=torch.complex64)

    y = model(X)
    print(y.dtype)

