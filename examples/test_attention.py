import torch
import vit_huy
import torchcvnn.nn as c_nn
import torchinfo


def test_attention():

    num_heads = 21
    head_dim = 16
    B, N, embed_dim = 11, 12, num_heads * head_dim
    input_size = (B, N, embed_dim)

    x = torch.randn((B, N, embed_dim), dtype=torch.complex64)

    # First model
    print("===" * 80)
    hatt = vit_huy.Block(embed_dim=embed_dim, 
                         hidden_dim=embed_dim, 
                         num_heads=num_heads)
    hy = hatt(x)
    print(hy.shape)
    torchinfo.summary(hatt, 
                      input_size=input_size, 
                      dtypes=[torch.complex64])

    # Torchcvnn
    print("===" * 80)
    oatt = c_nn.ViTLayer(num_heads=num_heads, 
                         hidden_dim=embed_dim, 
                         mlp_dim=embed_dim)
    oy = oatt(x)
    print(oy.shape)
    torchinfo.summary(oatt, 
                      input_size=input_size, 
                      dtypes=[torch.complex64])


if __name__ == "__main__":
    test_attention()

