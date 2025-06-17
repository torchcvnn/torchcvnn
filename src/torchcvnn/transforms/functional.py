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
from typing import Tuple, Dict
from types import ModuleType

# External imports
import torch
import numpy as np


def polsar_dict_to_array(x: np.ndarray | torch.Tensor | Dict[str, np.ndarray]) -> np.ndarray | torch.Tensor:
    """
    Convert a dictionary of numpy arrays to a stacked array.

    Args:
        x (np.ndarray | torch.Tensor | Dict[str, np.ndarray]): The input data. 
        It can be a single numpy array or PyTorch tensor, or a dictionary where keys are
        one of 'HH', 'HV', 'VH', 'VV' and values are arrays.

    Returns:
        np.ndarray | torch.Tensor: A stacked array from the dictionary's values if input is a dictionary,
        otherwise returns the input unchanged.
    
    Raises:
        AssertionError: If any key in the dictionary is not one of 'HH', 'HV', 'VH', 'VV'.
    """
    if isinstance(x, Dict):
        for k,v in x.items():
            assert k in ['HH', 'HV', 'VH', 'VV'], f'Invalid key {k} in input'
            assert isinstance(v, np.ndarray), "Values must be numpy arrays"
        return np.stack([
            v for v in x.values()
        ])
    return x


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


def log_normalize_amplitude(
    x: np.ndarray | torch.Tensor, 
    backend: ModuleType, 
    keep_phase: bool,
    min_value: float,
    max_value: float,
) -> np.ndarray | torch.Tensor:
    """
    Normalize the amplitude of a complex signal with logarithmic scaling.
    
    Args:
        x: Input array or tensor containing complex numbers. The type can be either numpy ndarray or PyTorch Tensor.
        backend: Module providing mathematical functions, allowing compatibility with numpy or PyTorch.
        keep_phase: Boolean indicating whether to retain the original phase of the input signal.
        max_value: Maximum amplitude value for normalization.
        min_value: Minimum amplitude value for normalization.

    Returns:
        A numpy ndarray or torch.Tensor containing the log-normalized amplitude, optionally with the original phase.
    """
    assert backend.__name__ in ["numpy", "torch"], "Backend must be numpy or torch"
    amplitude = backend.abs(x)
    phase = backend.angle(x)
    amplitude = backend.clip(amplitude, min_value, max_value)
    transformed_amplitude = (
        backend.log10(amplitude / min_value)
    ) / (np.log10(max_value / min_value))
    if keep_phase:
        return transformed_amplitude * backend.exp(1j * phase)
    else:
        return transformed_amplitude


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


def applyfft2_torch(x: torch.Tensor, dim: Tuple[int, ...] = (-2, -1)) -> torch.Tensor:
    """Apply 2D Fast Fourier Transform to image.
    
    Args:
        x (np.ndarray): Input array to apply FFT to
        dim (Tuple[int, ...]): Dimensions over which to compute the FFT. Default is (-2, -1).
        
    Returns:
        torch.Tensor: The Fourier transformed array
    """
    return torch.fft.fftshift(torch.fft.fft2(x, dim=dim), dim=dim)


def applyifft2_torch(x: torch.Tensor, dim: Tuple[int, ...] = (-2, -1)) -> torch.Tensor:
    """Apply 2D inverse Fast Fourier Transform to image.
    
    Args:
        x (torch.Tensor): Input tensor to apply IFFT to
        dim (Tuple[int, ...]): Dimensions over which to compute the IFFT. Default is (-2, -1).
        
    Returns:
        torch.Tensor: The inverse Fourier transformed array
    """
    return torch.fft.ifft2(torch.fft.ifftshift(x, dim=dim), dim=dim)


def get_padding(current_size: int, target_size: int) -> Tuple[int, ...]:
    """Calculate padding required to reach target size from current size.
    
    Calculates padding values for both sides of an axis to reach a target size.
    Handles both even and odd target sizes by adjusting padding distribution.
    
    Args:
        current_size (int): Current dimension size
        target_size (int): Desired dimension size after padding
        
    Returns:
        Tuple[int, ...]: Padding values for (before, after) positions
        
    Example:
        >>> get_padding(5, 7)  # Pad 5->7
        (1, 1)  # Add 1 padding on each side
        >>> get_padding(3, 6)  # Pad 3->6 (even target)
        (2, 1)  # More padding before for even targets
    
    Note:
        For even target sizes, the padding is distributed with one extra
        pad value before the content to maintain proper centering.
    """
    # Adjust offset for even-sized targets or odd-sized targets
    offset = 1 if target_size % 2 == 0 else 0
    # Calculate total padding needed
    pad_total = target_size - current_size
    # Calculate padding before, accounting for even-size offset
    pad_before = (pad_total + offset) // 2
    # Calculate padding after as remainder
    pad_after = pad_total - pad_before
    return pad_before, pad_after


def padifneeded(
    x: np.ndarray | torch.Tensor, 
    min_height: int, 
    min_width: int, 
    border_mode: str = "constant",
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
    top_pad, bottom_pad = get_padding(h, min_height)
    left_pad, right_pad = get_padding(w, min_width)
    padding = [
        top_pad, # top
        bottom_pad,  # bottom
        left_pad,  # left
        right_pad,  # right
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
    l_h = max(0, x.shape[1] // 2 - height // 2)
    l_w = max(0, x.shape[2] // 2 - width // 2)
    r_h = l_h + height
    r_w = l_w + width
    return x[:, l_h:r_h, l_w:r_w]


def rescale_intensity(image: np.ndarray, in_range: Tuple[float, float], out_range: Tuple[float, float]) -> np.ndarray:
    """Rescale the intensity of an image to a new range.
    
    Args:
        image (np.ndarray): Input image
        in_range (Tuple[float, float]): Input range (min, max) to scale from
        out_range (Tuple[float, float]): Output range (min, max) to scale to
        
    Returns:
        np.ndarray: Rescaled image
    """
    imin, imax = in_range
    omin, omax = out_range
    # Clip the input image to the input range
    image = np.clip(image, imin, imax)
    # Scale to [0, 1]
    if imin != imax:
        image = (image - imin) / (imax - imin)
        # Scale to output range
        return image * (omax - omin) + omin
    else:
        return np.clip(image, omin, omax)


def equalize(image: np.ndarray, plower: int = None, pupper: int = None) -> np.ndarray:
    """Automatically adjust contrast of the SAR image

    Args:
        image (np.ndarray): Image in complex
        plower (int, optional): lower percentile. Defaults to None.
        pupper (int, optional): upper percentile. Defaults to None.

    Returns:
        np.ndarray: Image equalized
    """
    image = np.log10(np.abs(image) + np.spacing(1))
    if not plower:
        vlower, vupper = np.percentile(image, (2, 98))
    else:
        vlower, vupper = np.percentile(image, (plower, pupper))
        
    return np.round(rescale_intensity(image, in_range=(vlower, vupper), out_range=(0, 1)) * 255).astype(np.uint8)
