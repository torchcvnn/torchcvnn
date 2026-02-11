import torch
import vit_huy
import torchcvnn.nn as c_nn
import torchinfo


def test_attention():

    num_heads = 21
    head_dim = 16
    B, S, E = 11, 12, num_heads * head_dim
    input_size = (B, S, E)

    x = torch.randn(input_size, dtype=torch.complex64)

    # First model
    print("===" * 80)
    hatt = vit_huy.Block(embed_dim=E, 
                         hidden_dim=E, 
                         num_heads=num_heads)
    hy = hatt(x)
    print(hy.shape)
    torchinfo.summary(hatt, 
                      input_size=input_size, 
                      dtypes=[torch.complex64])

    # Torchcvnn
    print("===" * 80)
    oatt = c_nn.ViTLayer(num_heads=num_heads, 
                         hidden_dim=E, 
                         mlp_dim=E,
                         norm_layer=c_nn.RMSNorm)
    oy = oatt(x)
    print(oy.shape)
    torchinfo.summary(oatt, 
                      input_size=input_size, 
                      dtypes=[torch.complex64])


if __name__ == "__main__":
    test_attention()

