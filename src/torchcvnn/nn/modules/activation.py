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
        batch_first: If `True`, then the input and output tensors are provided as (batch, seq, feature). Default `False` with tensors as (seq, batch, feature)


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
            # attn_output is (seq_len, batch_size, numè_features)

    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim: int = None,
        vdim: int = None,
        batch_first: bool = False,
        device: torch.device = None,
        dtype: torch.dtype = torch.complex64,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        if not self._qkv_same_embed_dim:
            self.q_proj_weight = torch.nn.parameter.Parameter(
                torch.empty((embed_dim, embed_dim), **factory_kwargs)
            )
            self.k_proj_weight = torch.nn.parameter.Parameter(
                torch.empty((embed_dim, self.kdim), **factory_kwargs)
            )
            self.v_proj_weight = torch.nn.parameter.Parameter(
                torch.empty((embed_dim, self.vdim), **factory_kwargs)
            )
            self.register_parameter("in_proj_weight", None)
        else:
            self.in_proj_weight = torch.nn.parameter.Parameter(
                torch.empty((3 * embed_dim, embed_dim), **factory_kwargs)
            )
            self.register_parameter("q_proj_weight", None)
            self.register_parameter("k_proj_weight", None)
            self.register_parameter("v_proj_weight", None)

        if bias:
            self.in_proj_bias = torch.nn.parameter.Parameter(
                torch.empty(3 * embed_dim, **factory_kwargs)
            )
        else:
            self.register_parameter("in_proj_bias", None)

        self.out_proj = torch.nn.Linear(
            embed_dim, embed_dim, bias=bias, **factory_kwargs
        )

        if add_bias_kv:
            self.bias_k = torch.nn.parameter.Parameter(
                torch.empty((1, 1, embed_dim), **factory_kwargs)
            )
            self.bias_v = torch.nn.parameter.Parameter(
                torch.empty((1, 1, embed_dim), **factory_kwargs)
            )
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn
        if bias:
            self.in_proj_bias = torch.nn.parameter.Parameter(
                torch.empty(3 * embed_dim, **factory_kwargs)
            )

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            complex_xavier_uniform_(self.in_proj_weight)
        else:
            complex_xavier_uniform_(self.q_proj_weight)
            complex_xavier_uniform_(self.k_proj_weight)
            complex_xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            torch.nn.init.constant_(self.in_proj_bias, 0.0)
            torch.nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            torch.nn.init.constant_(self.bias_k, 0.0)
        if self.bias_v is not None:
            torch.nn.init.constant_(self.bias_v, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Computes attention outputs using query, key and value embeddings.

        This function is adapted from torch.nn.MultiheadAttention to support complex valued tensors. It keeps the same
        signature but does not support yet key_padding_mask and attn_mask.
        """

        is_batched = query.dim() == 3

        if key_padding_mask is not None:
            raise NotImplementedError("key_padding_mask is not supported yet")
        # key_padding_mask = F._canonical_mask(
        #     mask=key_padding_mask,
        #     mask_name="key_padding_mask",
        #     other_type=F._none_or_dtype(attn_mask),
        #     other_name="attn_mask",
        #     target_type=query.dtype,  # Adapted because q is complex
        # )
        # But
        # F._canonical_mask raises an exception
        # AssertionError: only bool and floating types of key_padding_mask are supported

        if attn_mask is not None:
            raise NotImplementedError("attn_mask is not supported yet")
        # attn_mask = F._canonical_mask(
        #     mask=attn_mask,
        #     mask_name="attn_mask",
        #     other_type=None,
        #     other_name="",
        #     target_type=query.dtype,  # Adapted because q is complex
        #     check_other=False,
        # )

        if self.batch_first and is_batched:
            # These steps prevent multiple transpose on the same tensors
            # for example when using self-attention
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = (x.transpose(1, 0) for x in (query, key))
                    value = key
            else:
                query, key, value = (x.transpose(1, 0) for x in (query, key, value))

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = c_F.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight,
                k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
            )
        else:
            attn_output, attn_output_weights = c_F.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
            )
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights
