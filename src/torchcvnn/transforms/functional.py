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
from typing import Tuple

# External imports
import torch
import numpy as np


def is_chw_format(x: np.ndarray | torch.Tensor) -> bool:
    """Check if image is in CHW format."""
    if len(x.shape) != 3:
        raise ValueError("Image must be 3D array")
    if min(x.shape) != x.shape[0]:
        return False
    return True


def ensure_chw_format(x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Ensure image is in CHW format, convert if necessary."""
    if not isinstance(x, (np.ndarray, torch.Tensor)):
        raise TypeError("Image must be numpy array or torch tensor")
    if len(x.shape) != 3:
        raise ValueError("Image must be 3D array")
        
    if not is_chw_format(x):
        # Convert from HWC to CHW
        if isinstance(x, torch.Tensor):
            return x.permute(2, 0, 1)
        return x.transpose(2, 0, 1)
    return x


def applyfft2(x: np.ndarray, axis: Tuple[int, ...]) -> np.ndarray:
    """Apply 2D FFT to image."""
    return np.fft.fftshift(np.fft.fft2(x, axes=axis), axes=axis)


def applyifft2(x: np.ndarray, axis: Tuple[int, ...]) -> np.ndarray:
    """Apply 2D IFFT to image."""
    return np.fft.ifft2(np.fft.ifftshift(x, axes=axis), axes=axis)


def padifneeded(
    x: np.ndarray | torch.Tensor, 
    min_height: int, 
    min_width: int, 
    border_mode: str,
    pad_value: float = 0
) -> np.ndarray | torch.Tensor:
    """Pad image if smaller than desired size.
    
    Args:
        x: Input image with shape (C,H,W)
        min_height: Minimum height required
        min_width: Minimum width required
        border_mode: Padding mode ('constant', 'reflect', etc.)
        
    Returns:
        Padded image if needed, original otherwise
    """
    _, h, w = x.shape
    # Calculate padding sizes
    padding = [
        (min_height - h) // 2,  # top
        min_height - h - (min_height - h) // 2,  # bottom
        (min_width - w) // 2,  # left
        min_width - w - (min_width - w) // 2,  # right
    ]
    # Return original if no padding needed
    if all(p <= 0 for p in padding):
        return x
    
    padding = [max(0, p) for p in padding]
    if isinstance(x, np.ndarray):
        return np.pad(
            x,
            ((0, 0), (padding[0], padding[1]), (padding[2], padding[3])),
            mode=border_mode,
            constant_values=pad_value
        )
    return torch.nn.functional.pad(
        x,
        (padding[2], padding[3], padding[0], padding[1]),
        mode=border_mode,
        value=pad_value
    )