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


def check_input(x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Ensure image is in CHW format for 2D-tensors, convert if necessary.
    
    Args:
        x (np.ndarray or torch.Tensor): Input image to check/convert format
        
    Returns:
        np.ndarray or torch.Tensor: Image in CHW format
        
    Raises:
        TypeError: If input is not numpy array or torch tensor
        
    Example:
        >>> img = np.zeros((64, 64))  # HWC format
        >>> chw_img = check_input(img)  # Converts to (1, 64, 64)
    """
    if not isinstance(x, (np.ndarray, torch.Tensor)):
        raise TypeError("Element should be a numpy array or a tensor")
    if len(x.shape) == 2:
        return x[np.newaxis, :, :]
    return x


def applyfft2_np(x: np.ndarray, axis: Tuple[int, ...]) -> np.ndarray:
    """Apply 2D Fast Fourier Transform to image.
    
    Args:
        x (np.ndarray): Input array to apply FFT to
        axis (Tuple[int, ...]): Axes over which to compute the FFT
        
    Returns:
        np.ndarray: The Fourier transformed array
    """
    return np.fft.fftshift(np.fft.fft2(x, axes=axis), axes=axis)


def applyifft2_np(x: np.ndarray, axis: Tuple[int, ...]) -> np.ndarray:
    """Apply 2D inverse Fast Fourier Transform to image.
    
    Args:
        x (np.ndarray): Input array to apply IFFT to
        axis (Tuple[int, ...]): Axes over which to compute the IFFT
        
    Returns:
        np.ndarray: The inverse Fourier transformed array
    """
    return np.fft.ifft2(np.fft.ifftshift(x, axes=axis), axes=axis)


def applyfft2_torch(x: torch.Tensor, dim: Tuple[int, ...]) -> torch.Tensor:
    """Apply 2D Fast Fourier Transform to image.
    
    Args:
        x (np.ndarray): Input array to apply FFT to
        axis (Tuple[int, ...]): Axes over which to compute the FFT
        
    Returns:
        torch.Tensor: The Fourier transformed array
    """
    return torch.fft.fftshift(torch.fft.fft2(x, dim=dim), dim=dim)


def applyifft2_torch(x: torch.Tensor, dim: Tuple[int, ...]) -> torch.Tensor:
    """Apply 2D inverse Fast Fourier Transform to image.
    
    Args:
        x (torch.Tensor): Input tensor to apply IFFT to
        axis (Tuple[int, ...]): Axes over which to compute the IFFT
        
    Returns:
        torch.Tensor: The inverse Fourier transformed array
    """
    return torch.fft.ifft2(torch.fft.ifftshift(x, dim=dim), dim=dim)


def padifneeded(
    x: np.ndarray | torch.Tensor, 
    min_height: int, 
    min_width: int, 
    border_mode: str,
    pad_value: float = 0
) -> np.ndarray | torch.Tensor:
    """Pad image if smaller than desired size.

    This function pads an image with zeros if its dimensions are smaller than the specified
    minimum height and width. The padding is added equally on both sides where possible.

    Args:
        x (Union[np.ndarray, torch.Tensor]): Input image tensor/array with shape (C,H,W)
        min_height (int): Minimum required height after padding
        min_width (int): Minimum required width after padding 
        border_mode (str): Padding mode ('constant', 'reflect', 'replicate', etc.)
        pad_value (float): Value used for padding when border_mode is 'constant'. Default: 0

    Returns:
        Union[np.ndarray, torch.Tensor]: Padded image if dimensions were smaller than 
        minimum required, otherwise returns original image unchanged

    Example:
        >>> img = torch.randn(3, 50, 60)  # RGB image 50x60
        >>> padded = padifneeded(img, 64, 64, 'constant')  # Pads to 64x64
        >>> padded.shape
        torch.Size([3, 64, 64])
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


def center_crop(x: np.ndarray | torch.Tensor, height: int, width: int) -> np.ndarray | torch.Tensor:
    """
    Center crops an image to the specified dimensions.

    This function takes an image and crops it to the specified height and width, 
    centered around the middle of the image. If the requested dimensions are larger 
    than the image, it will use the maximum possible size.

    Args:
        x (Union[np.ndarray, torch.Tensor]): Input image tensor/array with shape (C, H, W)
        height (int): Desired height of the cropped image
        width (int): Desired width of the cropped image

    Returns:
        Union[np.ndarray, torch.Tensor]: Center cropped image with shape (C, height, width)

    Example:
        >>> img = torch.randn(3, 100, 100)  # RGB image 100x100
        >>> cropped = center_crop(img, 60, 60)  # Returns center 60x60 crop
        >>> cropped.shape
        torch.Size([3, 60, 60])
    """
    l_h = max(0, x.shape[0] // 2 - height // 2)
    l_w = max(0, x.shape[0] // 2 - width // 2)
    r_h = l_h + height
    r_w = l_w + width
    return x[:, l_h:r_h, l_w:r_w]