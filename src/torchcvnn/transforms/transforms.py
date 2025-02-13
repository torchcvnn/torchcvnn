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


class RandomPhase:
    """
    Transform a real tensor into a complex tensor, by applying a random phase to the tensor.
    """

    def __call__(self, tensor) -> torch.Tensor:
        phase = torch.rand_like(tensor, dtype=torch.float64) * 2 * torch.pi
        return (tensor * torch.exp(1j * phase)).to(torch.complex64)


class FFTResize(BaseTransform):
    """Resizes an input image in spectral domain with Fourier Transformations.

    This transform first applies a 2D FFT to the input array/tensor of shape CHW along specified axes,
    followed by padding or center cropping to achieve the target size, then applies
    an inverse FFT to go back to spatial domain. Optionally, it scales the output amplitudes to maintain energy consistency 
    between original and resized images.

    Args:
        size: Tuple[int, int]
            Target dimensions (height, width) for resizing.
        axis: Tuple[int, ...], optional
            The axes over which to apply FFT. Default is (-2, -1). For a array / tensor of shape CHW,
            it corresponds to the Height and Width axes.
        scale: bool, optional
            If True, scales the output amplitudes to maintain energy consistency with 
            respect to input size. Default is False.
        dtype: torch.dtype or numpy.dtype, optional
            Output data type. If None, maintains the input data type.
            For PyTorch tensors: torch.complex64 or torch.complex128
            For NumPy arrays: numpy.complex64 or numpy.complex128

    Returns:
        numpy.ndarray or torch.Tensor
            Resized image as a complex-valued array/tensor, maintaining shape (C, height, width).

    Examples:
        >>> transform = FFTResize((128, 128))
        >>> resized_image = transform(input_tensor)  # Resize to 128x128 using FFT

    Notes:
        - Input must be a multi-dimensional array/tensor of shape Channel x Height x Width.
        - Spectral domain resizing preserves frequency characteristics better than spatial interpolation
        - Operates on complex-valued data, preserving phase information
        - Memory efficient for large downsampling ratios
        - Based on the Fourier Transform properties of scaling and periodicity
        - The output is complex-valued due to the nature of FFT operations. If you are working with real-valued data,
        it is recommended to call ToReal after applying this transform.
    """
    def __init__(
        self, 
        size: Tuple[int, ...], 
        axis: Tuple[int, ...] = (-2, -1), 
        scale: bool = False, 
        dtype: Optional[str] = "complex64"
    ) -> None:
        if dtype is None or "complex" not in str(dtype):
            dtype = "complex64"
        
        super().__init__(dtype)
        assert isinstance(size, Tuple), "size must be a tuple"
        assert isinstance(axis, Tuple), "axis must be a tuple"
        self.height = size[0]
        self.width = size[1]
        self.axis = axis
        self.scale = scale

    def __call_numpy__(self, x: np.ndarray) -> np.ndarray:
        original_size = x.shape[1] * x.shape[2]
        target_size = self.height * self.width

        x = F.applyfft2_np(x, axis=self.axis)
        x = F.padifneeded(x, self.height, self.width)
        x = F.center_crop(x, self.height, self.width)
        x = F.applyifft2_np(x, axis=self.axis)

        if self.scale:
            return x * target_size / original_size
        return x.astype(self.np_dtype)
    
    def __call_torch__(self, x: torch.Tensor) -> torch.Tensor:
        original_size = x.shape[1] * x.shape[2]
        target_size = self.height * self.width

        x = F.applyfft2_torch(x, dim=self.axis)
        x = F.padifneeded(x, self.height, self.width)
        x = F.center_crop(x, self.height, self.width)
        x = F.applyifft2_torch(x, dim=self.axis)

        if self.scale:
            return x * target_size / original_size
        return x.to(self.torch_dtype)


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
