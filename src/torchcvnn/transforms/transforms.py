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
from types import ModuleType

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

    def __init__(self, dtype: str = None) -> None:
        if dtype is not None:
            assert isinstance(dtype, str), "dtype should be a string"
            assert dtype in [
                "float32",
                "float64",
                "complex64",
                "complex128",
            ], "dtype should be one of float32, float64, complex64, complex128"
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

    def __init__(
        self, min_value: float = 0.02, max_value: float = 40, keep_phase: bool = True
    ) -> None:
        self.min_value = min_value
        self.max_value = max_value
        self.keep_phase = keep_phase

    def __call_numpy__(self, x: np.ndarray) -> np.ndarray:
        return F.log_normalize_amplitude(
            x, np, self.keep_phase, self.min_value, self.max_value
        )

    def __call_torch__(self, x: torch.Tensor) -> torch.Tensor:
        return F.log_normalize_amplitude(
            x, torch, self.keep_phase, self.min_value, self.max_value
        )


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
        x = torch.stack([x.real, x.imag], dim=0)  # CHW -> 2CHW
        x = x.flatten(0, 1)  # 2CHW -> 2C*H*W
        return x

    def __call_numpy__(self, x: np.ndarray) -> np.ndarray:
        x = np.stack([x.real, x.imag], axis=0)  # CHW -> 2CHW
        x = x.reshape(-1, *x.shape[2:])  # 2CHW -> 2C*H*W
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
        pad_value: float = 0,
    ) -> None:
        self.min_height = min_height
        self.min_width = min_width
        self.border_mode = border_mode
        self.pad_value = pad_value

    def __call_numpy__(self, x: np.ndarray) -> np.ndarray:
        return F.padifneeded(
            x, self.min_height, self.min_width, self.border_mode, self.pad_value
        )

    def __call_torch__(self, x: torch.Tensor) -> torch.Tensor:
        return F.padifneeded(
            x, self.min_height, self.min_width, self.border_mode, self.pad_value
        )


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
        dtype: Optional[str] = "complex64",
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


class PolSAR(BaseTransform):
    """Handling Polarimetric Synthetic Aperture Radar (PolSAR) data channel conversions.
    This class provides functionality to convert between different channel representations of PolSAR data,
    supporting 1, 2, 3, and 4 output channel configurations. It can handle both NumPy arrays and PyTorch tensors.
    If inputs is a dictionnary of type {'HH': data1, 'VV': data2}, it will stack all values along axis 0 to form a CHW array.

    Args:
        out_channel (int): Desired number of output channels (1, 2, 3, or 4)

    Supported conversions:
        - 1 channel -> 1 channel: Identity
        - 2 channels -> 1 or 2 channels
        - 4 channels -> 1, 2, 3, or 4 channels where:
            - 1 channel: Returns first channel only
            - 2 channels: Returns [HH, VV] channels
            - 3 channels: Returns [HH, (HV+VH)/2, VV]
            - 4 channels: Returns all channels [HH, HV, VH, VV]

    Raises:
        ValueError: If the requested channel conversion is invalid or not supported

    Example:
        >>> transform = PolSAR(out_channel=3)
        >>> # For 4-channel input [HH, HV, VH, VV]
        >>> output = transform(input_data)  # Returns [HH, (HV+VH)/2, VV]

    Note:
        - Input data should have format Channels x Height x Width (CHW).
        - By default, PolSAR always return HH polarization if out_channel is 1.
    """

    def __init__(self, out_channel: int) -> None:
        self.out_channel = out_channel

    def _handle_single_channel(
        self, x: np.ndarray | torch.Tensor, out_channels: int
    ) -> np.ndarray | torch.Tensor:
        return x if out_channels == 1 else None

    def _handle_two_channels(
        self, x: np.ndarray | torch.Tensor, out_channels: int
    ) -> np.ndarray | torch.Tensor:
        if out_channels == 2:
            return x
        elif out_channels == 1:
            return x[0:1]
        return None

    def _handle_four_channels(
        self, x: np.ndarray | torch.Tensor, out_channels: int, backend: ModuleType
    ) -> np.ndarray | torch.Tensor:
        channel_maps = {
            1: lambda: x[0:1],
            2: lambda: backend.stack((x[0], x[3])),
            3: lambda: backend.stack((x[0], 0.5 * (x[1] + x[2]), x[3])),
            4: lambda: x,
        }
        return channel_maps.get(out_channels, lambda: None)()

    def _convert_channels(
        self, x: np.ndarray | torch.Tensor, out_channels: int, backend: ModuleType
    ) -> np.ndarray | torch.Tensor:
        handlers = {
            1: self._handle_single_channel,
            2: self._handle_two_channels,
            4: lambda x, o: self._handle_four_channels(x, o, backend),
        }
        result = handlers.get(x.shape[0], lambda x, o: None)(x, out_channels)
        if result is None:
            raise ValueError(
                f"Invalid conversion: {x.shape[0]} -> {out_channels} channels"
            )
        return result

    def __call_numpy__(self, x: np.ndarray) -> np.ndarray:
        return self._convert_channels(x, self.out_channel, np)

    def __call_torch__(self, x: torch.Tensor) -> torch.Tensor:
        return self._convert_channels(x, self.out_channel, torch)

    def __call__(
        self, x: np.ndarray | torch.Tensor | Dict[str, np.ndarray]
    ) -> np.ndarray | torch.Tensor:
        x = F.polsar_dict_to_array(x)
        return super().__call__(x)


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
