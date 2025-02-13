# MIT License

# Copyright (c) 2025 Quentin Gabot, Jeremy Fix, Huy Nguyen

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
from abc import ABC, abstractmethod
from typing import Tuple, Union, Optional, Dict
from types import NoneType, ModuleType

# External imports
import torch
import numpy as np
from PIL import Image

# Internal imports
import torchcvnn.transforms.functional as F


class LogAmplitude:
    """
    Transform the amplitude of a complex tensor to a log scale between a min and max value.

    After this transform, the phases are the same but the magnitude is log transformed and
    scaled in [0, 1]

    Arguments:
        min_value: The minimum value of the amplitude range to clip
        max_value: The maximum value of the amplitude range to clip
    """

    def __init__(self, min_value=0.02, max_value=40):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, tensor) -> torch.Tensor:
        new_tensor = []
        for idx, ch in enumerate(tensor):
            amplitude = torch.abs(ch)
            phase = torch.angle(ch)
            amplitude = torch.clip(amplitude, self.min_value, self.max_value)
            transformed_amplitude = (
                torch.log10(amplitude) - torch.log10(torch.tensor([self.min_value]))
            ) / (
                torch.log10(torch.tensor([self.max_value]))
                - torch.log10(torch.tensor([self.min_value]))
            )
            new_tensor.append(transformed_amplitude * torch.exp(1j * phase))
        return torch.as_tensor(np.stack(new_tensor), dtype=torch.complex64)


class Amplitude:
    """
    Transform a complex tensor into a real tensor, based on its amplitude.
    """

    def __call__(self, tensor) -> torch.Tensor:
        tensor = torch.abs(tensor).to(torch.float64)
        return tensor


class RealImaginary:
    """
    Transform a complex tensor into a real tensor, based on its real and imaginary parts.
    """

    def __call__(self, tensor) -> torch.Tensor:
        real = torch.real(tensor)
        imaginary = torch.imag(tensor)
        tensor_dual = torch.stack([real, imaginary], dim=0)
        tensor = tensor_dual.flatten(0, 1)  # concatenate real and imaginary parts
        return tensor
    

class ToReal:
    """Extracts the real part of a complex-valued input tensor.

    The `ToReal` transform takes either a numpy array or a PyTorch tensor containing complex numbers 
    and returns only their real parts. If the input is already real-valued, it remains unchanged.

    Returns:
        np.ndarray | torch.Tensor: A tensor with the same shape as the input but containing only 
                                  the real components of each element.
    
    Example:
        >>> to_real = ToReal()
        >>> output = to_real(complex_tensor)
    """
    def __call__(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        return x.real
    
    
class ToImaginary:
    """Extracts the imaginary part of a complex-valued input tensor.

    The `ToImaginary` transform takes either a numpy array or a PyTorch tensor containing complex numbers 
    and returns only their imaginary parts. If the input is already real-valued, it remains unchanged.

    Returns:
        np.ndarray | torch.Tensor: A tensor with the same shape as the input but containing only 
                                  the imaginary components of each element.
    
    Example:
        >>> to_imaginary = ToImaginary()
        >>> output = to_imaginary(complex_tensor)
    """
    def __call__(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        return x.imag


class RandomPhase:
    """
    Transform a real tensor into a complex tensor, by applying a random phase to the tensor.
    """

    def __call__(self, tensor) -> torch.Tensor:
        phase = torch.rand_like(tensor, dtype=torch.float64) * 2 * torch.pi
        return (tensor * torch.exp(1j * phase)).to(torch.complex64)
    
    
class FFT2(BaseTransform):
    """Applies 2D Fast Fourier Transform (FFT) to the input.
    This transform computes the 2D FFT along specified dimensions of the input array/tensor.
    It applies FFT2 and shifts zero-frequency components to the center.
    
    Args
        axis : Tuple[int, ...], optional
            The axes over which to compute the FFT. Default is (-2, -1).
        
    Returns
        numpy.ndarray or torch.Tensor
            The 2D Fourier transformed input with zero-frequency components centered.
            Output has the same shape as input.
    
    Notes
        - Transform is applied along specified dimensions (`axis`).
    """
    def __init__(self, axis: Tuple[int, ...] = (-2, -1)):
        self.axis = axis
    
    def __call_numpy__(self, x: np.ndarray) -> np.ndarray:
        return F.applyfft2_np(x, axis=self.axis)
    
    def __call_torch__(self, x: torch.Tensor) -> torch.Tensor:
        return F.applyfft2_torch(x, dim=self.axis)
    

class IFFT2(BaseTransform):
    """Applies 2D inverse Fast Fourier Transform (IFFT) to the input.
    This transform computes the 2D IFFT along the last two dimensions of the input array/tensor.
    It applies inverse FFT shift before IFFT2.
    
    Args
        axis : Tuple[int, ...], optional
            The axes over which to compute the FFT. Default is (-2, -1).

    Returns
        numpy.ndarray or torch.Tensor: 
            The inverse Fourier transformed input.
            Output has the same shape as input.
        
    Notes:
        - Transform is applied along specified dimensions (`axis`).
    """
    def __init__(self, axis: Tuple[int, ...] = (-2, -1)):
        self.axis = axis
        
    def __call_numpy__(self, x: np.ndarray) -> np.ndarray:
        return F.applyifft2_np(x, axis=self.axis)
    
    def __call_torch__(self, x: torch.Tensor) -> torch.Tensor:
        return F.applyifft2_torch(x, dim=self.axis)
    

class PadIfNeeded(BaseTransform):
    """Pad an image if its dimensions are smaller than specified minimum dimensions.

    This transform pads images that are smaller than given minimum dimensions by adding
    padding according to the specified border mode. The padding is added symmetrically
    on both sides to reach the minimum dimensions when possible. If the minimum required 
    dimension (height or width) is uneven, the right and the bottom sides will receive 
    an extra padding of 1 compared to the left and the top sides.

    Args:
        min_height (int): Minimum height requirement for the image
        min_width (int): Minimum width requirement for the image
        border_mode (str): Type of padding to apply ('constant', 'reflect', etc.). Default is 'constant'.
        pad_value (float): Value for constant padding (if applicable). Default is 0.

    Returns:
        np.ndarray | torch.Tensor: Padded image with dimensions at least min_height x min_width. 
        Original image if no padding is required.

    Example:
        >>> transform = PadIfNeeded(min_height=256, min_width=256)
        >>> padded_image = transform(small_image)  # Pads if image is smaller than 256x256
    """
    def __init__(
        self, 
        min_height: int,
        min_width: int,
        border_mode: str = "constant",
        pad_value: float = 0
    ) -> None:
        self.min_height = min_height
        self.min_width = min_width
        self.border_mode = border_mode
        self.pad_value = pad_value

    def __call_numpy__(self, x: np.ndarray) -> np.ndarray:
        return F.padifneeded(x, self.min_height, self.min_width, self.border_mode, self.pad_value)
    
    def __call_torch__(self, x: torch.Tensor) -> torch.Tensor:
        return F.padifneeded(x, self.min_height, self.min_width, self.border_mode, self.pad_value)


class CenterCrop(BaseTransform):
    """Center crops an input array/tensor to the specified size.

    This transform extracts a centered rectangular region from the input array/tensor
    with the specified dimensions. The crop is centered on both height and width axes.

    Args:
        height (int): Target height of the cropped output
        width (int): Target width of the cropped output

    Returns:
        np.ndarray | torch.Tensor: Center cropped array/tensor with shape (C, height, width), 
            where C is the number of channels

    Examples:
        >>> transform = CenterCrop(height=224, width=224)
        >>> output = transform(input_tensor)  # Center crops to 224x224

    Notes:
        - If input is smaller than crop size, it will return the original input
        - Crop is applied identically to all channels
        - Uses functional.center_crop() implementation for both numpy and torch
    """
    def __init__(self, height: int, width: int) -> None:
        self.height = height
        self.width = width

    def __call_numpy__(self, x: np.ndarray) -> np.ndarray:
        return F.center_crop(x, self.height, self.width)
    
    def __call_torch__(self, x: torch.Tensor) -> torch.Tensor:
        return F.center_crop(x, self.height, self.width)


class FFTResize:
    """
    Resize a complex tensor to a given size. The resize is performed in the Fourier
    domain by either cropping or padding the FFT2 of the input array/tensor.

    Arguments:
        size: The target size of the resized tensor.
    """

    def __init__(self, size):
        self.size = size

    def __call__(
        self, array: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:

        is_torch = False
        if isinstance(array, torch.Tensor):
            is_torch = True
            array = array.numpy()

        real_part = array.real
        imaginary_part = array.imag

        def zoom(array):
            # Computes the 2D FFT of the array and center the zero frequency component
            array = np.fft.fftshift(np.fft.fft2(array))
            original_size = array.shape

            # Either center crop or pad the array to the target size
            target_size = self.size
            if array.shape[0] < target_size[0]:
                # Computes top and bottom padding
                top_pad = (target_size[0] - array.shape[0] + 1) // 2
                bottom_pad = target_size[0] - array.shape[0] - top_pad
                array = np.pad(array, ((top_pad, bottom_pad), (0, 0)))
            elif array.shape[0] > target_size[0]:
                top_crop = array.shape[0] // 2 - target_size[0] // 2
                bottom_crop = top_crop + target_size[0]
                array = array[top_crop:bottom_crop, :]

            if array.shape[1] < target_size[1]:
                left_pad = (target_size[1] - array.shape[1] + 1) // 2
                right_pad = target_size[1] - array.shape[1] - left_pad
                array = np.pad(array, ((0, 0), (left_pad, right_pad)))
            elif array.shape[1] > target_size[1]:
                left_crop = array.shape[1] // 2 - target_size[1] // 2
                right_crop = left_crop + target_size[1]
                array = array[:, left_crop:right_crop]

            # Computes the inverse 2D FFT of the array
            array = np.fft.ifft2(np.fft.ifftshift(array))
            scale = (target_size[0] * target_size[1]) / (
                original_size[0] * original_size[1]
            )

            return scale * array

        if len(array.shape) == 2:
            # We have a two dimensional tensor
            resized_real = zoom(real_part)
            resized_imaginary = zoom(imaginary_part)
        else:
            # We have three dimensions and therefore
            # apply the resize to each channel iteratively
            # We assume the first dimension is the channel
            resized_real = []
            resized_imaginary = []
            for real, imaginary in zip(real_part, imaginary_part):
                resized_real.append(zoom(real))
                resized_imaginary.append(zoom(imaginary))
            resized_real = np.stack(resized_real)
            resized_imaginary = np.stack(resized_imaginary)

        resized_array = resized_real + 1j * resized_imaginary

        # Convert the resized tensor back to a torch tensor if necessary
        if is_torch:
            resized_array = torch.as_tensor(resized_array)

        return resized_array


class SpatialResize:
    """
    Resize a complex tensor to a given size. The resize is performed in the image space
    using a Bicubic interpolation.

    Arguments:
        size: The target size of the resized tensor.
    """

    def __init__(self, size):
        self.size = size

    def __call__(
        self, array: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:

        is_torch = False
        if isinstance(array, torch.Tensor):
            is_torch = True
            array = array.numpy()

        real_part = array.real
        imaginary_part = array.imag

        def zoom(array):
            # Convert the numpy array to a PIL image
            image = Image.fromarray(array)

            # Resize the image
            image = image.resize((self.size[1], self.size[0]))

            # Convert the PIL image back to a numpy array
            array = np.array(image)

            return array

        if len(array.shape) == 2:
            # We have a two dimensional tensor
            resized_real = zoom(real_part)
            resized_imaginary = zoom(imaginary_part)
        else:
            # We have three dimensions and therefore
            # apply the resize to each channel iteratively
            # We assume the first dimension is the channel
            resized_real = []
            resized_imaginary = []
            for real, imaginary in zip(real_part, imaginary_part):
                resized_real.append(zoom(real))
                resized_imaginary.append(zoom(imaginary))
            resized_real = np.stack(resized_real)
            resized_imaginary = np.stack(resized_imaginary)

        resized_array = resized_real + 1j * resized_imaginary

        # Convert the resized tensor back to a torch tensor if necessary
        if is_torch:
            resized_array = torch.as_tensor(resized_array)

        return resized_array


class PolSARtoTensor:
    """
    Transform a PolSAR image into a 3D torch tensor.
    """

    def __call__(self, element: Union[np.ndarray, dict]) -> torch.Tensor:
        if isinstance(element, np.ndarray):
            assert len(element.shape) == 3, "Element should be a 3D numpy array"
            if element.shape[0] == 3:
                return self._create_tensor(element[0], element[1], element[2])
            if element.shape[0] == 2:
                return self._create_tensor(element[0], element[1])
            elif element.shape[0] == 4:
                return self._create_tensor(
                    element[0], (element[1] + element[2]) / 2, element[3]
                )

        elif isinstance(element, dict):
            if len(element) == 3:
                return self._create_tensor(element["HH"], element["HV"], element["VV"])
            elif len(element) == 2:
                if "HH" in element:
                    return self._create_tensor(element["HH"], element["HV"])
                elif "VV" in element:
                    return self._create_tensor(element["HV"], element["VV"])
                else:
                    raise ValueError(
                        "Dictionary should contain keys HH, HV, VV or HH, VV"
                    )
            elif len(element) == 4:
                return self._create_tensor(
                    element["HH"], (element["HV"] + element["VH"]) / 2, element["VV"]
                )
        else:
            raise ValueError("Element should be a numpy array or a dictionary")

    def _create_tensor(self, *channels) -> torch.Tensor:
        return torch.as_tensor(
            np.stack(channels, axis=-1).transpose(2, 0, 1),
            dtype=torch.complex64,
        )


class Unsqueeze:
    """
    Add a dimension to a tensor.

    Arguments:
        dim: The dimension of the axis/dim to extend
    """

    def __init__(self, dim):
        self.dim = dim

    def __call__(self, element: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Apply the transformation by adding a dimension to the input tensor.
        """
        if isinstance(element, np.ndarray):
            element = np.expand_dims(element, axis=self.dim)
        elif isinstance(element, torch.Tensor):
            element = element.unsqueeze(dim=self.dim)


class ToTensor:
    """
    Convert a numpy array to a tensor.
    """

    def __call__(self, element: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(element, np.ndarray):
            return torch.as_tensor(element)
        elif isinstance(element, torch.Tensor):
            return element
        else:
            raise ValueError("Element should be a numpy array or a tensor")
