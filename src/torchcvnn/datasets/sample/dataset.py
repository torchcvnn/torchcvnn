# MIT License

# Copyright (c) 2024 Jeremy Fix

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
import requests
import pathlib
import logging
from typing import Tuple

# External imports
import torch
from torch.utils.data import Dataset
import scipy.io
import tqdm
import numpy as np

# Local imports
from .filelist import filelist

SAMPLE_base_link = "https://github.com/benjaminlewis-afrl/SAMPLE_dataset_public/raw/refs/heads/master/mat_files/"


class SAMPLE(Dataset):
    """
    The SAMPLE dataset is made partly from real data provided by MSTAR and partly from synthetic data.

    The dataset is public and will be downloaded if requested and missing on drive.

    It is made of 10 classes of military vehicles: 2s1, bmp2, btr70, m1, m2, m35, m548, m60, t72, zsu23

    Arguments:
        rootdir (str): Path to the root directory where the dataset is stored or will be downloaded
        transform (torchvision.transforms.Compose): A list of torchvision transforms to apply to the complex image
        download (bool): Whether to download the data if missing on disk

    Note:
        An example usage :

        .. code-block:: python

            import torchcvnn
            from torchcvnn.datasets import SAMPLE

            transform = v2.Compose(
                transforms=[v2.ToImage(), v2.Resize(128), v2.CenterCrop(128)]
            )
            dataset = SAMPLE(
                rootdir, transform=transform, download=True
            )
            X, y = dataset[0]

        Displayed below are some examples drawn randomly from SAMPLE. To plot them, we extracted
        only the magnitude of the signals although the data are indeed complex valued.

        .. image:: ../assets/datasets/SAMPLE.png
           :alt: Samples from MSTAR
           :width: 60%

    """

    def __init__(self, rootdir: str, transform=None, download: bool = False):
        super().__init__()
        self.rootdir = pathlib.Path(rootdir)
        self.transform = transform

        self.class_names = list(filelist["real"].keys())

        # We look into rootdir if the data are available
        self.tot_num_samples = 0
        for cl in self.class_names:
            self.tot_num_samples += len(filelist["real"][cl])
            self.tot_num_samples += len(filelist["synth"][cl])

        self.data_files = {}
        self.num_data_files = {}
        self.tot_num_data_files = 0

        pbar = tqdm.tqdm(total=self.tot_num_samples)
        for cl in self.class_names:
            for mode in ["real", "synth"]:
                for filename in filelist[mode][cl]:
                    filepath = self.rootdir / cl / filename
                    if not filepath.exists():
                        if download:
                            url = f"{SAMPLE_base_link}{mode}/{cl}/{filename}"
                            self.download_file(url, filepath)
                        else:
                            raise FileNotFoundError(f"{filepath} not found")
                    pbar.update(1)
            self.data_files[cl] = list((self.rootdir / cl).glob("*.mat"))
            self.num_data_files[cl] = len(self.data_files[cl])
            self.tot_num_data_files += self.num_data_files[cl]

    def download_file(self, url: str, filepath: pathlib.Path):
        """
        Download a file from an URL and save it on disk

        Args:
            url (str): URL to download the file from
            filepath (pathlib.Path): Path to save the file
        """
        logging.debug(f"Downloading {url} to {filepath}")
        # Ensure the target directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Donwload and save the file on disk
        response = requests.get(url)
        with open(filepath, "wb") as fh:
            fh.write(response.content)

    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset
        """
        return self.tot_num_data_files

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Return a sample from the dataset
        """
        if index >= self.tot_num_data_files:
            raise IndexError

        # We look for the class from which the sample will be taken
        for key in self.data_files.keys():
            if index < self.num_data_files[key]:
                break
            index -= self.num_data_files[key]

        filename = self.data_files[key][index]
        logging.debug(f"Reading SAMPLE file : {filename}")

        data = scipy.io.loadmat(filename)

        # Below are the keys available in the mat files
        # dict_keys(['__header__', '__version__', '__globals__', 'aligned', 'azimuth', 'bandwidth', 'center_freq', 'complex_img', 'complex_img_unshifted', 'elevation', 'explanation', 'range_pixel_spacing', 'range_resolution', 'source_mstar_file', 'target_name', 'taylor_weights', 'xrange_pixel_spacing', 'xrange_resolution']

        meta = {
            k: data[k]
            for k in [
                "azimuth",
                "elevation",
                "bandwidth",
                "center_freq",
                "range_pixel_spacing",
                "range_resolution",
                "xrange_pixel_spacing",
                "xrange_resolution",
            ]
        }

        complex_img = data["complex_img"][:, :, np.newaxis]

        class_idx = self.class_names.index(key)

        if self.transform is not None:
            complex_img = self.transform(complex_img)

        return complex_img, class_idx, meta
