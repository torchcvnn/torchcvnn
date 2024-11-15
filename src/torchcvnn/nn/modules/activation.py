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


# External imports
import torch
import torch.nn as nn


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

    :math:`CReLU(z) = ReLU(Re[z]) + ReLU(Im[z])j`

    Only the quadrant where both `Re[z]` and `Im[z]` are negative is projected to
    :math:`0`. Otherwise either the real and/or the imaginary part is preserved.

    """

    def __init__(self) -> None:
        super().__init__(nn.ReLU)


class CPReLU(IndependentRealImag):
    """
    Applies a PReLU independently on both the real and imaginary parts

    :math:`CPReLU(z) = PReLU(Re[z]) + PReLU(Im[z])j`
    """

    def __init__(self) -> None:
        super().__init__(nn.PReLU)


class CELU(IndependentRealImag):
    """
    Applies a ELU independently on both the real and imaginary parts

    Not to confuse with `torch.nn.CELU`. For the complex equivalent of
    `torch.nn.CELU`, see [torchcvnn.nn.modules.activation.CCELU][]

    :math:`CELU(z) = ELU(Re[z]) + ELU(Im[z])j`
    """

    def __init__(self) -> None:
        super().__init__(nn.ELU)


class CCELU(IndependentRealImag):
    """
    Applies a CELU independently on both the real and imaginary parts

    :math:`CCELU(z) = CELU(Re[z]) + CELU(Im[z])j`
    """

    def __init__(self) -> None:
        super().__init__(nn.CELU)


class CGELU(IndependentRealImag):
    """
    Applies a GELU independently on both the real and imaginary parts

    :math:`CGELU(z) = GELU(Re[z]) + GELU(Im[z])j`
    """

    def __init__(self) -> None:
        super().__init__(nn.GELU)


class CSigmoid(IndependentRealImag):
    """
    Applies a Sigmoid independently on both the real and imaginary parts

    as used in Nitta Tohru. An extension of the back-propagation algorithm to complex numbers. Neural Networks, 10(9):1391–1415, November 1997.

    :math:`CSigmoid(z) = Sigmoid(Re[z]) + Sigmoid(Im[z])j`

    where the real valued sigmoid is applied in the right hand side terms.
    """

    def __init__(self) -> None:
        super().__init__(nn.Sigmoid)


class CTanh(IndependentRealImag):
    """
    Applies a Tanh independently on both the real and imaginary parts

    :math:`CTanh(z) = Tanh(Re[z]) + Tanh(Im[z])j`

    where the real valued sigmoid is applied in the right hand side terms.
    """

    def __init__(self) -> None:
        super().__init__(nn.Tanh)


class zReLU(nn.Module):
    r"""
    Applies a zReLU

    :math:`zReLU(z) = \begin{cases} z & \mbox{if } Re[z] > 0 \mbox{ and } Im[z] > 0\\ 0 & \mbox{otherwise}  \end{cases}`

    All the quadrant where both `Re[z]` and `Im[z]` are non negative are
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
    Applies a zReLU

    :math:`zLeakyReLU(z) = \begin{cases} z & \mbox{if } Re[z] > 0 \mbox{ and } Im[z] > 0\\ a.z & \mbox{otherwise}  \end{cases}`

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
    Extracts the magnitude of the complex input: maps to :math:`\mathbb{R}`

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
    Applies a ReLU with parametric offset on the amplitude, keeping the phase unchanged.

    :math:`Cardioid(z) = \frac{1+\cos(\theta)}{2} z`

    As proposed by Virtue et al. (2019). For real numbers, e.g. :math:`\theta \in \{0,
    \pi\}`, it reduces to the ReLU :

    :math:`\forall r \in \mathbb{R}, \theta \in \{0, \pi\}, Cardiod(r e^{\theta j}) = Relu(r)`
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
        return 0.5 * (1 + torch.cos(z.angle())) * z
