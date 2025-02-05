# MIT License

# Copyright (c) 2025 Huy Nguyen

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

# External imports
import torch
import numpy as np


def is_chw_format(image: np.ndarray | torch.Tensor) -> bool:
    """Check if image is in CHW format."""
    if len(image.shape) != 3:
        raise ValueError("Image must be 3D array")
    if min(image.shape) != image.shape[0]:
        return False
    return True


def ensure_chw_format(image: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Ensure image is in CHW format, convert if necessary."""
    if not isinstance(image, (np.ndarray, torch.Tensor)):
        raise TypeError("Image must be numpy array or torch tensor")
    if len(image.shape) != 3:
        raise ValueError("Image must be 3D array")
        
    if not is_chw_format(image):
        # Convert from HWC to CHW
        if isinstance(image, torch.Tensor):
            return image.permute(2, 0, 1)
        return image.transpose(2, 0, 1)
    return image


def applyfft2(image: np.ndarray) -> np.ndarray:
    """Apply 2D FFT to image."""
    return np.fft.fftshift(np.fft.fft2(image, axes=(0, 1)), axes=(0, 1))


def applyifft2(image: np.ndarray) -> np.ndarray:
    """Apply 2D IFFT to image."""
    return np.fft.ifft2(np.fft.ifftshift(image, axes=(0, 1)), axes=(0, 1))