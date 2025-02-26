# MIT License

# Copyright (c) 2024-2025 Jeremy Fix, Huy Nguyen

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

# External imports
import torch
import numpy as np

# Local imports
import torchcvnn.transforms as transforms


def test_fft_resize_ndarray():
    # Create a random complex tensor
    tensor = np.random.rand(100, 100) + 1j * np.random.rand(100, 100)

    # Resize the tensor
    target_size = (59, 49)
    spatial_resize = transforms.FFTResize(target_size)
    resized_tensor = spatial_resize(tensor)

    assert (resized_tensor.shape[-2], resized_tensor.shape[-1]) == target_size
    assert type(resized_tensor) == np.ndarray
    assert resized_tensor.dtype in [np.complex64, np.complex128]

    # Resize the tensor
    target_size = (123, 121)
    spatial_resize = transforms.FFTResize(target_size)
    resized_tensor = spatial_resize(tensor)
    assert (resized_tensor.shape[-2], resized_tensor.shape[-1]) == target_size
    assert type(resized_tensor) == np.ndarray
    assert resized_tensor.dtype in [np.complex64, np.complex128]


def test_fft_resize_tensor():
    # Create a random complex tensor
    tensor = np.random.rand(100, 100) + 1j * np.random.rand(100, 100)
    tensor = torch.as_tensor(tensor)

    # Resize the tensor
    target_size = (59, 49)
    spatial_resize = transforms.FFTResize(target_size)
    resized_tensor = spatial_resize(tensor)

    assert (resized_tensor.shape[-2], resized_tensor.shape[-1]) == target_size
    assert type(resized_tensor) == torch.Tensor
    assert resized_tensor.dtype in [torch.complex64, torch.complex128]

    # Resize the tensor
    target_size = (123, 121)
    spatial_resize = transforms.FFTResize(target_size)
    resized_tensor = spatial_resize(tensor)
    assert (resized_tensor.shape[-2], resized_tensor.shape[-1]) == target_size
    assert type(resized_tensor) == torch.Tensor
    assert resized_tensor.dtype in [torch.complex64, torch.complex128]


def test_spatial_resize_ndarray():
    # Create a random complex tensor
    tensor = np.random.rand(100, 100) + 1j * np.random.rand(100, 100)

    # Resize the tensor
    target_size = (59, 48)
    spatial_resize = transforms.SpatialResize(target_size)
    resized_tensor = spatial_resize(tensor)

    assert resized_tensor.shape == target_size
    assert type(resized_tensor) == np.ndarray
    assert resized_tensor.dtype == np.complex64

    # Resize the tensor
    target_size = (123, 121)
    spatial_resize = transforms.SpatialResize(target_size)
    resized_tensor = spatial_resize(tensor)

    assert resized_tensor.shape == target_size
    assert type(resized_tensor) == np.ndarray
    assert resized_tensor.dtype == np.complex64


def test_spatial_resize_tensor():
    # Create a random complex tensor
    tensor = np.random.rand(100, 100) + 1j * np.random.rand(100, 100)
    tensor = torch.as_tensor(tensor)

    # Resize the tensor
    target_size = (50, 50)
    spatial_resize = transforms.SpatialResize(target_size)
    resized_tensor = spatial_resize(tensor)

    assert resized_tensor.shape == target_size
    assert type(resized_tensor) == torch.Tensor
    assert resized_tensor.dtype == torch.complex64

    # Resize the tensor
    target_size = (121, 121)
    spatial_resize = transforms.SpatialResize(target_size)
    resized_tensor = spatial_resize(tensor)

    assert resized_tensor.shape == target_size
    assert type(resized_tensor) == torch.Tensor
    assert resized_tensor.dtype == torch.complex64


def on_rgb_img():
    """
    Test the resize from an image
    """
    from PIL import Image
    import matplotlib.pyplot as plt

    im = Image.open("img_rgb.jpg")

    array = np.array(im)
    array = array - 1j * array
    array = np.transpose(array, (2, 0, 1)) / 255.0

    # 487, 565
    target_size = (123, 456)

    spatial_resize = transforms.SpatialResize(target_size)
    resized1 = spatial_resize(array)

    fft_resize = transforms.FFTResize(target_size)
    resized2 = fft_resize(array)

    plt.subplot(1, 3, 1)
    plt.imshow(array.real.transpose(1, 2, 0), clim=(0, 1))
    plt.title("Original")
    plt.subplot(1, 3, 2)
    plt.imshow(resized1.real.transpose(1, 2, 0), clim=(0, 1))
    plt.title("Spatial resize")
    plt.subplot(1, 3, 3)
    plt.imshow(resized2.real.transpose(1, 2, 0), clim=(0, 1))
    plt.title("FFT resize")
    plt.show()


def on_bw_img():
    """
    Test the resize from an image
    """
    from PIL import Image
    import matplotlib.pyplot as plt

    im = Image.open("img_bw.jpg")

    array = np.array(im)
    array = array - 1j * array
    array = array / 255.0

    target_size = (123, 456)

    spatial_resize = transforms.SpatialResize(target_size)
    resized1 = spatial_resize(array)

    fft_resize = transforms.FFTResize(target_size)
    resized2 = fft_resize(array)

    plt.subplot(1, 3, 1)
    plt.imshow(array.real, cmap="gray", clim=(0, 1))
    plt.title("Original")
    plt.subplot(1, 3, 2)
    plt.imshow(resized1.real, cmap="gray", clim=(0, 1))
    plt.title("Spatial resize")
    plt.subplot(1, 3, 3)
    plt.imshow(resized2.real, cmap="gray", clim=(0, 1))
    plt.title("FFT resize")
    plt.show()


if __name__ == "__main__":
    test_fft_resize_ndarray()
    test_fft_resize_tensor()
    test_spatial_resize_ndarray()
    test_spatial_resize_tensor()
    on_rgb_img()
    on_bw_img()
