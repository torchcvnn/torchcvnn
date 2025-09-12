# MIT License

# Copyright (c) 2024 Quentin Gabot

# The following is an adaptation of the initialization code from
# PyTorch library for the complex-valued neural networks

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


class ComplexMSELoss(nn.modules.loss._Loss):
    r"""A custom implementation of Mean Squared Error (MSE) Loss for complex-valued inputs.

    This loss function is designed to compute the mean squared error between two sets of complex-valued tensors. Unlike the standard MSE loss which directly computes the square of the difference between predicted values and true values for real numbers, this implementation adapts to complex numbers by calculating the squared magnitude of the difference between the complex predicted and true values before averaging.

    For complex-valued numbers, the MSE is defined as:
        MSE = mean(|y_true - y_pred|^2)
    where |y_true - y_pred| denotes the magnitude of the complex difference between the true and predicted values. This formula ensures that both the real and imaginary parts of the complex numbers are considered in the loss calculation.

    For real-valued numbers, the standard MSE formula is:
        MSE = mean((y_true - y_pred)^2)
    where (y_true - y_pred)^2 is the square of the difference between the true and predicted values.

    Parameters:
        size_average (bool, optional): Deprecated (use reduction). By default, the losses are averaged over each loss element in the batch. Note: when (reduce is False), size_average is ignored. Default is None.
        reduce (bool, optional): Deprecated (use reduction). By default, the losses are averaged or summed over observations for each minibatch depending on reduction. Note: when (reduce is False), size_average is ignored. Default is None.
        reduction (str, optional): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. 'none': no reduction will be applied, 'mean': the sum of the output will be divided by the number of elements in the output, 'sum': the output will be summed. Default is 'mean'.

    Inputs:
        y_pred (torch.Tensor): The predicted values, can be a complex tensor.
        y_true (torch.Tensor): The ground truth values, must have the same shape as the predicted values, can be a complex tensor.

    Returns:
        torch.Tensor: The calculated mean squared error loss.

    Examples:
        >>> loss = ComplexMSELoss()
        >>> y_pred = torch.tensor([1+1j, 2+2j], dtype=torch.complex64)
        >>> y_true = torch.tensor([1+2j, 2+3j], dtype=torch.complex64)
        >>> output = loss(y_pred, y_true)
        >>> print(output)
    """

    def __init__(self, size_average=None, reduce=None, reduction: str = "mean") -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # Calculate Mean Square Loss for complex numbers
        mse = torch.square(torch.abs(y_true - y_pred))
        if self.reduction == "mean":
            return torch.mean(mse)
        elif self.reduction == "sum":
            return torch.sum(mse)
        elif self.reduction == "none":
            return mse
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")
