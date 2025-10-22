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

def test_polsar_transform_four_channels():
    # Create a random complex tensor
    tensor_numpy = np.random.rand(4, 100, 100) + 1j * np.random.rand(4, 100, 100)
    tensor = torch.as_tensor(tensor_numpy)

    polsar_transform_one_channel = transforms.PolSAR(out_channel=1)
    out = polsar_transform_one_channel(tensor_numpy)
    assert out.shape == (1, 100, 100)
    out = polsar_transform_one_channel(tensor)
    assert out.shape == (1, 100, 100)

    polsar_transform_two_channels = transforms.PolSAR(out_channel=2)
    out = polsar_transform_two_channels(tensor_numpy)
    assert out.shape == (2, 100, 100)    
    out = polsar_transform_two_channels(tensor)
    assert out.shape == (2, 100, 100)

    polsar_transform_three_channels = transforms.PolSAR(out_channel=3)
    out = polsar_transform_three_channels(tensor_numpy)
    assert out.shape == (3, 100, 100)
    out = polsar_transform_three_channels(tensor)
    assert out.shape == (3, 100, 100)

    polsar_transform_four_channels = transforms.PolSAR(out_channel=4)
    out = polsar_transform_four_channels(tensor_numpy)
    assert out.shape == (4, 100, 100)
    out = polsar_transform_four_channels(tensor)
    assert out.shape == (4, 100, 100)    

def test_polsar_transform_three_channels():
    # Create a random complex tensor
    tensor_numpy = np.random.rand(3, 100, 100) + 1j * np.random.rand(3, 100, 100)
    tensor = torch.as_tensor(tensor_numpy)

    polsar_transform_one_channel = transforms.PolSAR(out_channel=1)
    out = polsar_transform_one_channel(tensor_numpy)
    assert out.shape == (1, 100, 100)
    out = polsar_transform_one_channel(tensor)
    assert out.shape == (1, 100, 100)

    polsar_transform_two_channels = transforms.PolSAR(out_channel=2)
    out = polsar_transform_two_channels(tensor_numpy)
    assert out.shape == (2, 100, 100)    
    out = polsar_transform_two_channels(tensor)
    assert out.shape == (2, 100, 100)

    polsar_transform_three_channels = transforms.PolSAR(out_channel=3)
    out = polsar_transform_three_channels(tensor_numpy)
    assert out.shape == (3, 100, 100)
    out = polsar_transform_three_channels(tensor)
    assert out.shape == (3, 100, 100)

def polsar_transform_two_channels():
    # Create a random complex tensor
    tensor_numpy = np.random.rand(2, 100, 100) + 1j * np.random.rand(2, 100, 100)
    tensor = torch.as_tensor(tensor_numpy)

    polsar_transform_one_channel = transforms.PolSAR(out_channel=1)
    out = polsar_transform_one_channel(tensor_numpy)
    assert out.shape == (1, 100, 100)
    out = polsar_transform_one_channel(tensor)
    assert out.shape == (1, 100, 100)

    polsar_transform_two_channels = transforms.PolSAR(out_channel=2)
    out = polsar_transform_two_channels(tensor_numpy)
    assert out.shape == (2, 100, 100)    
    out = polsar_transform_two_channels(tensor)
    assert out.shape == (2, 100, 100)

def polsar_transform_one_channel():
    # Create a random complex tensor
    tensor_numpy = np.random.rand(1, 100, 100) + 1j * np.random.rand(1, 100, 100)
    tensor = torch.as_tensor(tensor_numpy)

    polsar_transform_one_channel = transforms.PolSAR(out_channel=1)
    out = polsar_transform_one_channel(tensor_numpy)
    assert out.shape == (1, 100, 100)
    out = polsar_transform_one_channel(tensor)
    assert out.shape == (1, 100, 100)

def test_normalize():
    # Create complex Gaussian data that matches the provided per-channel means and covariances
    rng = np.random.default_rng(0)
    C, H, W = 2, 100, 100
    means = [[0.0, 0.0], [1.0, 1.0]]
    covs = [[[1.0, 0.0], [0.0, 1.0]], [[2.0, 0.0], [0.0, 2.0]]]

    # Generate per-channel complex Gaussian samples with shape (C, H, W)
    samples = np.zeros((C, H, W), dtype=np.complex64)
    for c in range(C):
        mv = rng.multivariate_normal(means[c], covs[c], size=H * W)
        real = mv[:, 0].reshape(H, W)
        imag = mv[:, 1].reshape(H, W)
        samples[c] = real + 1j * imag

    tensor_numpy = samples
    tensor = torch.as_tensor(tensor_numpy)

    normalize_transform = transforms.Normalize(means, covs)

    # Numpy input
    out_np = normalize_transform(tensor_numpy)
    assert out_np.shape == (C, H, W)

    # For each channel, check mean ~ 0 and covariance ~ identity
    for c in range(C):
        vals = out_np[c].reshape(-1)
        mean_real = vals.real.mean()
        mean_imag = vals.imag.mean()
        # sampling error over H*W samples -> use a realistic tolerance
        assert np.allclose([mean_real, mean_imag], [0.0, 0.0], atol=5e-2)

        stacked = np.vstack([vals.real.flatten(), vals.imag.flatten()])
    cov_est = np.cov(stacked, bias=True)
    assert np.allclose(cov_est, np.eye(2), atol=3e-2)

    # Torch input
    out_t = normalize_transform(tensor)
    assert out_t.shape == (C, H, W)

    for c in range(C):
        vals = out_t[c].cpu().numpy().reshape(-1)
        mean_real = vals.real.mean()
        mean_imag = vals.imag.mean()
        # sampling error over H*W samples -> use a realistic tolerance
        assert np.allclose([mean_real, mean_imag], [0.0, 0.0], atol=5e-2)

        stacked = np.vstack([vals.real.flatten(), vals.imag.flatten()])
    cov_est = np.cov(stacked, bias=True)
    assert np.allclose(cov_est, np.eye(2), atol=3e-2)

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
    test_polsar_transform_four_channels()
    test_polsar_transform_three_channels()
    polsar_transform_two_channels()
    polsar_transform_one_channel()
    test_normalize()
