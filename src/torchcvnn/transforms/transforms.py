# MIT License

# Copyright (c) 2025 Quentin Gabot

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
from typing import Union

# External imports
import torch
import numpy as np
from PIL import Image


class LogAmplitude:
    """
    Transform the amplitude of a complex tensor to a log scale between a min and max value.
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

    def __init__(self):
        pass

    def __call__(self, tensor) -> torch.Tensor:
        tensor = torch.abs(tensor).to(torch.float64)
        return tensor


class RealImaginary:
    """
    Transform a complex tensor into a real tensor, based on its real and imaginary parts.
    """

    def __init__(self):
        pass

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
        return tensor * torch.exp(1j * phase)


class SpatialResize:
    """
    Resize a complex tensor to a given size.
    """

    def __init__(self, size):
        self.size = size

    def __call__(
        self, array: Union[np.array, torch.tensor]
    ) -> Union[np.array, torch.Tensor]:

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
            image = image.resize(self.size)

            # Convert the PIL image back to a numpy array
            array = np.array(image)

            return array

        resized_real = zoom(real_part)
        resized_imaginary = zoom(imaginary_part)

        resized_array = resized_real + 1j * resized_imaginary

        # Convert the resized tensor back to a torch tensor if necessary
        if is_torch:
            resized_array = torch.as_tensor(resized_array)

        return resized_array


class PolSARtoTensor:
    """
    Transform a PolSAR image into a 3D torch tensor.
    """

    def __call__(self, element) -> torch.Tensor:
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
    """

    def __init__(self, dim):
        self.dim = dim

    def __call__(self, element) -> torch.Tensor:
        if isinstance(element, np.ndarray):
            element = np.expand_dims(element, axis=self.dim)
        elif isinstance(element, torch.Tensor):
            element = element.unsqueeze(dim=self.dim)


class ToTensor:
    """
    Convert a numpy array to a tensor.
    """

    def __call__(self, element) -> torch.Tensor:
        if isinstance(element, np.ndarray):
            return torch.as_tensor(element)
        elif isinstance(element, torch.Tensor):
            return element
        else:
            raise ValueError("Element should be a numpy array or a tensor")


def test_spatial_resize():
    """
    Test the SpatialResize transform with both
    a np.array and torch.tensor.
    """

    # Create a random complex tensor
    tensor = np.random.rand(100, 100) + 1j * np.random.rand(100, 100)
    tensor = torch.as_tensor(tensor)

    # Resize the tensor
    spatial_resize = SpatialResize((50, 50))
    resized_tensor = spatial_resize(tensor)

    # Check the shape of the resized tensor
    assert resized_tensor.shape == (3, 50, 50)

    # Check the type of the resized tensor
    assert isinstance(resized_tensor, torch.Tensor)

    # Convert the tensor to a numpy array
    tensor = tensor.numpy()

    # Resize the tensor
    resized_tensor = spatial_resize(tensor)

    # Check the shape of the resized tensor
    assert resized_tensor.shape == (3, 50, 50)

    # Check the type of the resized tensor
    assert isinstance(resized_tensor, np.ndarray)


if __name__ == "__main__":
    test_spatial_resize()
