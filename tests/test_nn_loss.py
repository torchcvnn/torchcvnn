# MIT License

# Copyright (c) 2024 Quentin Gabot

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
import numpy as np

# Local imports
import torchcvnn.nn as c_nn


def test_complex_mse_loss():
    reduction_modes = ["mean", "sum", "none"]
    for mode in reduction_modes:
        loss = c_nn.ComplexMSELoss(reduction=mode)
        y_pred = torch.tensor([1 + 1j, 2 + 2j], dtype=torch.complex64)
        y_true = torch.tensor([1 + 2j, 2 + 3j], dtype=torch.complex64)
        output = loss(y_pred, y_true)
        expected_outputs = {
            "mean": torch.tensor(1.0),
            "sum": torch.tensor(2.0),
            "none": torch.tensor([1.0, 1.0]),
        }
        assert torch.allclose(output, expected_outputs[mode]), f"Failed for reduction mode: {mode}"


if __name__ == "__main__":
    test_complex_mse_loss()
