import torch
import vit_huy
import torchcvnn.nn as c_nn


def test_attention():

    num_heads = 21
    head_dim = 16
    B, N, embed_dim = 11, 12, num_heads * head_dim

    x = torch.randn((B, N, embed_dim), dtype=torch.complex64)

    # First model
    hatt = vit_huy.Attention(embed_dim, num_heads)
    hy = hatt(x)
    print(hy.shape)

    # Torchcvnn
    oatt = c_nn.ViTLayer(num_heads=num_heads, 
                         hidden_dim=embed_dim, 
                         mlp_dim=embed_dim)
    oy = oatt(x)
    print(oy.shape)


if __name__ == "__main__":
    test_attention()

