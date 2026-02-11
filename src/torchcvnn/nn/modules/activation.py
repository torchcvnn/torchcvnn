# MIT License

# Copyright (c) 2023 Jérémie Levi, Victor Dhédin, Jeremy Fix

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
from typing import Optional

# External imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# Local imports
from torchcvnn.nn import functional as c_F
from .initialization import complex_xavier_uniform_


class IndependentRealImag(nn.Module):
    """
    Generic module to apply a real valued activation function independently
    on both the real and imaginary part

    Arguments:
        fact: A nn.Module name of a real valued activation function
    """

    def __init__(self, fact: nn.Module):
        super().__init__()
        self.act_real = fact()
        self.act_imag = fact()

    def forward(self, z: torch.tensor) -> torch.tensor:
        """
        Performs the forward pass

        Arguments:
            z: the input tensor on which to apply the activation function
        """
        return self.act_real(z.real) + self.act_imag(z.imag) * 1j


class CReLU(IndependentRealImag):
    """
    Applies a ReLU independently on both the real and imaginary parts

    :math:`CReLU(z) = ReLU(\\Re[z]) + ReLU(\\Im[z])j`

    Only the quadrant where both `\\Re[z]` and `\\Im[z]` are negative is projected to
    :math:`0`. Otherwise either the real and/or the imaginary part is preserved.

    """

    def __init__(self) -> None:
        super().__init__(nn.ReLU)


class CPReLU(IndependentRealImag):
    """
    Applies a PReLU independently on both the real and imaginary parts

    :math:`CPReLU(z) = PReLU(\\Re[z]) + PReLU(\\Im[z])j`
    """

    def __init__(self) -> None:
        super().__init__(nn.PReLU)


class CELU(IndependentRealImag):
    """
    Applies a ELU independently on both the real and imaginary parts

    Not to confuse with `torch.nn.CELU`. For the complex equivalent of
    :external:py:class:`torch.nn.CELU`, see :class:`torchcvnn.nn.modules.activation.CCELU`

    :math:`CELU(z) = ELU(\\Re[z]) + ELU(\\Im[z])j`
    """

    def __init__(self) -> None:
        super().__init__(nn.ELU)


class CCELU(IndependentRealImag):
    """
    Applies a CELU independently on both the real and imaginary parts

    :math:`CCELU(z) = CELU(\\Re[z]) + CELU(\\Im[z])j`
    """

    def __init__(self) -> None:
        super().__init__(nn.CELU)


class CGELU(IndependentRealImag):
    """
    Applies a GELU independently on both the real and imaginary parts

    :math:`CGELU(z) = GELU(\\Re[z]) + GELU(\\Im[z])j`
    """

    def __init__(self) -> None:
        super().__init__(nn.GELU)


class CSigmoid(IndependentRealImag):
    """
    Applies a Sigmoid independently on both the real and imaginary parts

    as used in Nitta Tohru. An extension of the back-propagation algorithm to complex numbers. Neural Networks, 10(9):1391–1415, November 1997.

    :math:`CSigmoid(z) = Sigmoid(\\Re[z]) + Sigmoid(\\Im[z])j`

    where the real valued sigmoid is applied in the right hand side terms.
    """

    def __init__(self) -> None:
        super().__init__(nn.Sigmoid)


class CTanh(IndependentRealImag):
    """
    Applies a Tanh independently on both the real and imaginary parts

    :math:`CTanh(z) = \\tanh(\\Re[z]) + \\tanh(\\Im[z])j`

    where the real valued sigmoid is applied in the right hand side terms.
    """

    def __init__(self) -> None:
        super().__init__(nn.Tanh)


class zReLU(nn.Module):
    r"""
    Applies a zReLU

    :math:`zReLU(z) = \begin{cases} z & \mbox{if } \Re[z] > 0 \mbox{ and } \Im[z] > 0\\ 0 & \mbox{otherwise}  \end{cases}`

    All the quadrant where both :math:`\Re[z]` and :math:`\Im[z]` are non negative are
    projected to :math:`0`. In other words, only one quadrant is preserved.
    """

    def __init__(self):
        super().__init__()

    def forward(self, z: torch.Tensor):
        """
        Performs the forward pass.

        Arguments:
            z: the input tensor on which to apply the activation function
        """
        pos_real = z.real > 0
        pos_img = z.imag > 0
        return z * pos_real * pos_img


class zAbsReLU(nn.Module):
    r"""
    Applies a zAbsReLU

    :math:`zAbsReLU(z) = \begin{cases} z & \mbox{if } |z| \geq a\\ 0 & \mbox{otherwise}  \end{cases}`

    This cancels all the complex plane in the circle of radius :math:`a`, where :math:`a` is
    trainable.
    """

    def __init__(self):
        super().__init__()
        self.a = torch.nn.parameter.Parameter(
            data=torch.Tensor([1.0]), requires_grad=True
        )

    def forward(self, z: torch.Tensor):
        """
        Performs the forward pass.

        Arguments:
            z: the input tensor on which to apply the activation function
        """
        mask = z.abs() < self.a
        return z * mask


class zLeakyReLU(nn.Module):
    r"""
    Applies a zLeakyReLU

    :math:`zLeakyReLU(z) = \begin{cases} z & \mbox{if } \Re[z] > 0 \mbox{ and } \Im[z] > 0\\ a.z & \mbox{otherwise}  \end{cases}`

    """

    def __init__(self):
        super().__init__()
        self.a = torch.nn.parameter.Parameter(
            data=torch.Tensor([0.2]), requires_grad=True
        )

    def forward(self, z: torch.Tensor):
        """
        Performs the forward pass.

        Arguments:
            z: the input tensor on which to apply the activation function
        """
        pos_real = z.real > 0
        pos_img = z.imag > 0
        return z * pos_real * pos_img + self.a * (z * ~(pos_real * pos_img))


class Mod(nn.Module):
    r"""
    Extracts the magnitude of the complex input. It maps to :math:`\mathbb{R}`

    :math:`Mod(z) = |z|`

    This activation function allows to go from complex values to real
    values.

    """

    def __init__(self):
        super().__init__()

    def forward(self, z: torch.Tensor):
        """
        Performs the forward pass.

        Arguments:
            z: the input tensor on which to apply the activation function
        """
        return torch.abs(z)


class modReLU(nn.Module):
    r"""
    Applies a ReLU with parametric offset on the amplitude, keeping the phase unchanged.

    :math:`modReLU(z) = ReLU(|z| + b) e^{j \theta}`
    """

    def __init__(self):
        super().__init__()
        self.b = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float), True)

    def forward(self, z: torch.Tensor):
        """
        Performs the forward pass.

        Arguments:
            z: the input tensor on which to apply the activation function
        """
        return nn.functional.relu(z.abs() + self.b) * torch.exp(1j * z.angle())


class Cardioid(nn.Module):
    r"""
    The cardioid activation function as proposed by Virtue et al. (2019) is given by :

    :math:`Cardioid(z) = \frac{1+\cos(\theta)}{2} z`

    For real numbers, e.g. :math:`\theta \in \{0, \pi\}`, it reduces to the ReLU :

    :math:`\forall r \in \mathbb{R}, \theta \in \{0, \pi\}, Cardioid(r e^{j \theta}) = ReLU(r) e^{j \theta} = ReLU(r)`
    """

    def __init__(self):
        super().__init__()

    def forward(self, z: torch.Tensor):
        """
        Performs the forward pass.

        Arguments:
            z: the input tensor on which to apply the activation function
        """
        return 0.5 * (1 + torch.cos(z.angle())) * z


class MultiheadAttention(nn.Module):
    """

    This class is adapted from torch.nn.MultiheadAttention to support complex valued tensors.

    Allows the model to jointly attend to information from different
    representation subspaces as described in the paper
    [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

    .. math::
        \mbox{MultiHead}(Q, K, V) = [head_1, \dots, head_h] W^O

    where :math:`head_i = \mbox{Attention}(Q W^Q_i, KW^K_i, VW^V_i)`


    This implementation is based on the paper **Building blocks for a complex-valued
    transformer architecture**. Florian Eilers, Xiaoyi Jiang. 2023. In International Conference on Acoustics, Speech,
    and Signal Processing (ICASSP).

    Attention is defined as follows:

    .. math::

        \mbox{Attention}(Q, K, V) = \sigma(\\Re[\\frac{Q K^H}{\sqrt{d_k}}])V

    Arguments:
        embed_dim: Total dimension of the model.
        num_heads: Number of parallel heads. Note that `embed_dim` will be split accross `num_heads` (i.e. each head will have dimension `embed_dim // num_heads`)
        dropout: Dropout probability on `attn_output_weights`. Default: `0.0`
        kdim: Total number of features for keys. Default `None` which uses `kdim=embed_dim`
        vdim: Total number of features for keys. Default `None` which uses `vdim=embed_dim`
        batch_first: If `True`, then the input (query, key, value) and output tensors (attn_outputs) are provided as (batch, seq, feature). Default `False` with tensors as (seq, batch, feature)


    Example:

        .. code-block:: python

            import torchcvnn as c_nn
            import torch

            nhead = 8
            seq_len = 10
            batch_size = 32
            num_features = 512

            multihead_attn = c_nn.MultiheadAttention(embed_dim=num_features, num_heads=nhead)
            src = torch.rand(seq_len, batch_size, num_features, dtype=torch.complex64)
            attn_output, attn_output_weights = multihead_attn(src, src, src)
            # attn_output is (seq_len, batch_size, num_features)

    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = False,
        device: torch.device = None,
        dtype: torch.dtype = torch.complex64,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.in_proj_weight = torch.nn.parameter.Parameter(
            torch.empty((3 * embed_dim, embed_dim), **factory_kwargs)
        )

        if bias:
            self.in_proj_bias = torch.nn.parameter.Parameter(
                torch.empty(3 * embed_dim, **factory_kwargs)
            )
        else:
            self.register_parameter("in_proj_bias", None)

        self.out_proj = torch.nn.Linear(
            embed_dim, embed_dim, bias=bias, **factory_kwargs
        )

        self._reset_parameters()

    def _reset_parameters(self):
        complex_xavier_uniform_(self.in_proj_weight)

        if self.in_proj_bias is not None:
            torch.nn.init.constant_(self.in_proj_bias, 0.0)
            torch.nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        need_weights: bool = True,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Computes attention outputs using query, key and value embeddings.

        This function is adapted from torch.nn.MultiheadAttention to support complex valued tensors. 

        Shape:
            Inputs:
            - query: :math:`(T, E)` or :math:`(T, B, E)` (``batch_first=False``) or :math:`(B, T, E) (``batch_first=True``), where T is the target sequence length, B is the batch size, E is the embedding dimension
            - key: :math:`(S, E)` or :math:`(S, B, E)` (``batch_first=False``) or :math:`(B, S, E) (``batch_first=True``), where S is the source sequence length, B is the batch size, E is the embedding dimension.
            - value: :math:`(S, E)` or :math:`(S, B, E)` (``batch_first=False``) or :math:`(B, S, E) (``batch_first=True``), where S is the source sequence length, B is the batch size, E is the embedding dimension.

        """

        is_batched = query.dim() == 3

        if self.batch_first and is_batched:
            # In this case, query is (B, T, E), key is (B, S, E) and value is (B, S, E)

            # These steps prevent multiple transpose on the same tensors
            # for example when using self-attention
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = (x.transpose(1, 0) for x in (query, key))
                    value = key
            else:
                query, key, value = (x.transpose(1, 0) for x in (query, key, value)) # (T, B, E), (S, B, E), (S, B, E)
    
        print(f"Query shapes : {query.shape}") # T, B, E
        print(f"Key shapes : {key.shape}") # S, B, E
        print(f"Value shapes : {value.shape}") # S, B, E

        attn_output, attn_output_weights = c_F.multi_head_attention_forward(
            query,
            key,
            value,
            self.embed_dim,
            self.num_heads,
            self.in_proj_weight,
            self.in_proj_bias,
            self.dropout,
            self.out_proj.weight,
            self.out_proj.bias,
            training=self.training,
            need_weights=need_weights,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )
        # attn_output is (T, E) or (T, B, E)
        # attn_output_weights is (T, S) or (B, T, S) (already batch_first)
        if is_batched and self.batch_first:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights
