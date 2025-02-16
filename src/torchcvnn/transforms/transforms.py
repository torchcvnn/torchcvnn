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


class BaseTransform(ABC):
    """Abstract base class for transforms that can handle both numpy arrays and PyTorch tensors.
    This class serves as a template for implementing transforms that can be applied to both numpy arrays
    and PyTorch tensors while maintaining consistent behavior. 
    Inputs must be in CHW (Channel, Height, Width) format. If inputs have only 2 dimensions (Height, Width),
    they will be converted to (1, Height, Width).
    
    Args:
        dtype (str, optional): Data type to convert inputs to. Must be one of:
            'float32', 'float64', 'complex64', 'complex128'. If None, no type conversion is performed.
            Default: None.
            
    Raises:
        AssertionError: If dtype is not a string or not one of the allowed types.
        ValueError: If input is neither a numpy array nor a PyTorch tensor.
        
    Methods:
        __call__(x): Apply the transform to the input array/tensor.
        __call_numpy__(x): Abstract method to implement numpy array transform.
        __call_torch__(x): Abstract method to implement PyTorch tensor transform.
        
    Example:
        >>> class MyTransform(BaseTransform):
        >>>     def __call_numpy__(self, x):
        >>>         # Implement numpy transform
        >>>         pass
        >>>     def __call_torch__(self, x):
        >>>         # Implement torch transform
        >>>         pass
        >>> transform = MyTransform(dtype='float32')
        >>> output = transform(input_data)  # Works with both numpy arrays and torch tensors
    """
    def __init__(self, dtype: str | NoneType = None) -> None:
        if dtype is not None:
            assert isinstance(dtype, str), "dtype should be a string"
            assert dtype in ["float32", "float64", "complex64", "complex128"], "dtype should be one of float32, float64, complex64, complex128"
            self.np_dtype = getattr(np, dtype)
            self.torch_dtype = getattr(torch, dtype)
    
    def __call__(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Apply transform to input."""
        x = F.check_input(x)
        if isinstance(x, np.ndarray):
            return self.__call_numpy__(x)
        elif isinstance(x, torch.Tensor):
            return self.__call_torch__(x)

    @abstractmethod
    def __call_numpy__(self, x: np.ndarray) -> np.ndarray:
        """Apply transform to numpy array."""
        raise NotImplementedError
    
    @abstractmethod
    def __call_torch__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply transform to torch tensor."""
        raise NotImplementedError


class LogAmplitude(BaseTransform):
    """This transform applies a logarithmic scaling to the amplitude/magnitude of complex values
    while optionally preserving the phase information. The amplitude is first clipped to 
    [min_value, max_value] range, then log10-transformed and normalized to [0,1] range.

    The transformation follows these steps:
    1. Extract amplitude and phase from complex input
    2. Clip amplitude between min_value and max_value 
    3. Apply log10 transform and normalize to [0,1]
    4. Optionally recombine with original phase

    Args:
        min_value (int | float, optional): Minimum amplitude value for clipping. 
            Values below this will be clipped up. Defaults to 0.02.
        max_value (int | float, optional): Maximum amplitude value for clipping.
            Values above this will be clipped down. Defaults to 40.
        keep_phase (bool, optional): Whether to preserve phase information.
            If True, returns complex output with transformed amplitude and original phase.
            If False, returns just the transformed amplitude. Defaults to True.
    Returns:
        np.ndarray | torch.Tensor: Transformed tensor with same shape as input.
            If keep_phase=True: Complex tensor with log-scaled amplitude and original phase
            If keep_phase=False: Real tensor with just the log-scaled amplitude
    Example:
        >>> transform = LogAmplitude(min_value=0.01, max_value=100)
        >>> output = transform(input_tensor)  # Transforms amplitudes to log scale [0,1]
    Note:
        The transform works with both NumPy arrays and PyTorch tensors through
        separate internal implementations (__call_numpy__ and __call_torch__).
    """
    def __init__(self, min_value: float = 0.02, max_value: float = 40, keep_phase: bool = True) -> None:
        self.min_value = min_value
        self.max_value = max_value
        self.keep_phase = keep_phase

    def __call_numpy__(self, x: np.ndarray) -> np.ndarray:
        return F.log_normalize_amplitude(x, np, self.keep_phase, self.min_value, self.max_value)
        
    def __call_torch__(self, x: torch.Tensor) -> torch.Tensor:
        return F.log_normalize_amplitude(x, torch, self.keep_phase, self.min_value, self.max_value)


class Amplitude(BaseTransform):
    """Transform a complex-valued tensor into its amplitude/magnitude.

    This transform computes the absolute value (magnitude) of complex input data,
    converting complex values to real values.

    Args:
        dtype (str): Data type for the output ('float32', 'float64', etc)

    Returns:
        np.ndarray | torch.Tensor: Real-valued tensor containing the amplitudes,
            with same shape as input but real-valued type specified by dtype.
    """
    def __init__(self, dtype: str) -> None:
        super().__init__(dtype)

    def __call_torch__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.abs(x).to(self.torch_dtype)
    
    def __call_numpy__(self, x: np.ndarray) -> np.ndarray:
        return np.abs(x).astype(self.np_dtype)


class RealImaginary(BaseTransform):
    """Transform a complex-valued tensor into its real and imaginary components.

    This transform separates a complex-valued tensor into its real and imaginary parts,
    stacking them along a new channel dimension. The output tensor has twice the number
    of channels as the input.

    Returns:
        np.ndarray | torch.Tensor: Real-valued tensor containing real and imaginary parts,
            with shape (2*C, H, W) where C is the original number of channels.
    """
    def __call_torch__(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.stack([x.real, x.imag], dim=0) # CHW -> 2CHW
        x = x.flatten(0, 1) # 2CHW -> 2C*H*W
        return x
    
    def __call_numpy__(self, x: np.ndarray) -> np.ndarray:
        x = np.stack([x.real, x.imag], axis=0) # CHW -> 2CHW
        x = x.reshape(-1, *x.shape[2:]) # 2CHW -> 2C*H*W
        return x
    

class RandomPhase(BaseTransform):
    """Randomly phase-shifts complex-valued input data.
    This transform applies a random phase shift to complex-valued input tensors/arrays by 
    multiplying the input with exp(j*phi), where phi is uniformly distributed in [0, 2π] 
    or [-π, π] if centering is enabled.
    Args:
        dtype : str
            Data type for the output. Must be one of the supported complex dtypes.
        centering : bool, optional. 
            If True, centers the random phase distribution around 0 by subtracting π from 
            the generated phases. Default is False.
    Returns
        torch.Tensor or numpy.ndarray
            Phase-shifted complex-valued data with the same shape as input.

    Examples
        >>> transform = RandomPhase(dtype='complex64')
        >>> x = torch.ones(3,3, dtype=torch.complex64)
        >>> output = transform(x)  # Applies random phase shifts

    Notes
        - Input data must be complex-valued
        - The output maintains the same shape and complex dtype as input
        - Phase shifts are uniformly distributed in:
            - [0, 2π] when centering=False
            - [-π, π] when centering=True
    """
    def __init__(self, dtype: str, centering: bool = False) -> None:
        super().__init__(dtype)
        self.centering = centering

    def __call_torch__(self, x: torch.Tensor) -> torch.Tensor:
        phase = torch.rand_like(x) * 2 * torch.pi
        if self.centering:
            phase = phase - torch.pi
        return (x * torch.exp(1j * phase)).to(self.torch_dtype)
    
    def __call_numpy__(self, x: np.ndarray) -> np.ndarray:
        phase = np.random.rand(*x.shape) * 2 * np.pi
        if self.centering:
            phase = phase - np.pi
        return (x * np.exp(1j * phase)).astype(self.np_dtype)
    

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


class Unsqueeze(BaseTransform):
    """Add a singleton dimension to the input array/tensor.

    This transform inserts a new axis at the specified position, increasing 
    the dimensionality of the input by one.

    Args:
        dim (int): Position where new axis should be inserted.

    Returns:
        np.ndarray | torch.Tensor: Input with new singleton dimension added.
            Shape will be same as input but with a 1 inserted at position dim.

    Example:
        >>> transform = Unsqueeze(dim=0) 
        >>> x = torch.randn(3,4)  # Shape (3,4)
        >>> y = transform(x)      # Shape (1,3,4)
    """
    def __init__(self, dim: int) -> None:
        self.dim = dim

    def __call_numpy__(self, x: np.ndarray) -> np.ndarray:
        return np.expand_dims(x, axis=self.dim)
    
    def __call_torch__(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(dim=self.dim)


class ToTensor(BaseTransform):
    """Converts numpy array or torch tensor to torch tensor of specified dtype.
    This transform converts input data to a PyTorch tensor with the specified data type.
    It handles both numpy arrays and existing PyTorch tensors as input.

    Args:
        dtype (str): Target data type for the output tensor. Should be one of PyTorch's
            supported dtype strings (e.g. 'float32', 'float64', 'int32', etc.)

    Returns:
        torch.Tensor: The converted tensor with the specified dtype.
        
    Example:
        >>> transform = ToTensor(dtype='float32')
        >>> x_numpy = np.array([1, 2, 3])
        >>> x_tensor = transform(x_numpy)  # converts to torch.FloatTensor
        >>> x_existing = torch.tensor([1, 2, 3], dtype=torch.int32)
        >>> x_converted = transform(x_existing)  # converts to torch.FloatTensor
    """
    def __init__(self, dtype: str) -> None:
        super().__init__(dtype)

    def __call_numpy__(self, x: np.ndarray) -> np.ndarray:
        return torch.as_tensor(x, dtype=self.torch_dtype)
    
    def __call_torch__(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(self.torch_dtype)
