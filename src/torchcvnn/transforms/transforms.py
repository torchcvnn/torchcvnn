import torch
import numpy as np
from scipy.ndimage import zoom


MIN_VALUE = 0.02
MAX_VALUE = 40


class LogAmplitudeTransform:
    def __init__(self, min_value=MIN_VALUE, max_value=MAX_VALUE):
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


class AmplitudeTransform:
    def __init__(self):
        pass

    def __call__(self, tensor) -> torch.Tensor:
        tensor = torch.abs(tensor).to(torch.float64)
        return tensor


class RealImaginaryTransform:
    def __init__(self):
        pass

    def __call__(self, tensor) -> torch.Tensor:
        real = torch.real(tensor)
        imaginary = torch.imag(tensor)
        tensor_dual = torch.stack([real, imaginary], dim=0)
        tensor = tensor_dual.flatten(0, 1)  # concatenate real and imaginary parts
        return tensor


class RandomPhaseTransform:

    def __call__(self, tensor) -> torch.Tensor:
        phase = torch.rand_like(tensor, dtype=torch.float64) * 2 * torch.pi
        return tensor * torch.exp(1j * phase)


class ComplexResizeTransform:
    def __init__(self, size):
        self.size = size

    def __call__(self, tensor) -> torch.Tensor:
        real_part = tensor.real
        imaginary_part = tensor.imag

        zoom_factors = (self.size / tensor.shape[0], self.size / tensor.shape[1])

        resized_real = zoom(real_part, zoom_factors, order=3)  # Cubic interpolation
        resized_imaginary = zoom(
            imaginary_part, zoom_factors, order=3
        )  # Cubic interpolation

        resized_array = resized_real + 1j * resized_imaginary

        return resized_array


class PolSARtoArrayTransform:
    def __init__(self):
        pass

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
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, element) -> torch.Tensor:
        if isinstance(element, np.ndarray):
            element = np.expand_dims(element, axis=self.dim)
        elif isinstance(element, torch.Tensor):
            element = element.unsqueeze(dim=self.dim)


class ToTensor:

    def __call__(self, element) -> torch.Tensor:
        if isinstance(element, np.ndarray):
            return torch.as_tensor(element)
        elif isinstance(element, torch.Tensor):
            return element
        else:
            raise ValueError("Element should be a numpy array or a tensor")
