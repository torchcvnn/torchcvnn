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
import os

# External imports
import numpy as np
from torch.utils.data import Dataset


class S1SLC(Dataset):
    r"""
    The Polarimetric SAR dataset with the labels provided by
    https://ieee-dataport.org/open-access/s1slccvdl-complex-valued-annotated-single-look-complex-sentinel-1-sar-dataset-complex

    We expect the data to be already downloaded and available on your drive.

    Arguments:
        root: the top root dir where the data are expected. The data should be organized as follows: Sao Paulo/HH.npy, Sao Paulo/HV.npy, Sao Paulo/Labels.npy, Houston/HH.npy, Houston/HV.npy, Houston/Labels.npy, Chicago/HH.npy, Chicago/HV.npy, Chicago/Labels.npy
        transform : the transform applied the cropped image
        lazy_loading : if True, the data is loaded only when requested. If False, the data is loaded at the initialization of the dataset.

    Note:
        An example usage :

        .. code-block:: python

            import torchcvnn
            from torchcvnn.datasets import S1SLC

            def transform(patches):
                # If you wish, you could filter out some polarizations
                # S1SLC provides the dual HH, HV polarizations
                patches = [np.abs(patchi) for _, patchi in patches.items()]
                return np.stack(patches)

            dataset = S1SLC(rootdir, transform=transform
            X, y = dataset[0]

    """

    def __init__(self, root, transform=None, lazy_loading=False):
        if lazy_loading is True:
            raise DeprecationWarning(
                "Lazy loading is no longer supported for S1SLC dataset."
            )

        self.transform = transform
        # Get list of subfolders in the root path
        subfolders = [
            os.path.join(root, name)
            for name in os.listdir(root)
            if os.path.isdir(os.path.join(root, name))
        ]

        self.data = {}
        self.classes = set()

        for subfolder in subfolders:
            # Define paths to the .npy files
            hh_path = os.path.join(subfolder, "HH.npy")
            hv_path = os.path.join(subfolder, "HV.npy")
            labels_path = os.path.join(subfolder, "Labels.npy")

            # Load the .npy files (using a memory map for memory efficiency)
            hh = np.load(hh_path, mmap_mode="r")
            hv = np.load(hv_path, mmap_mode="r")

            # Load labels and convert to 0-indexed
            labels = np.load(labels_path, mmap_mode="r")
            labels = labels.astype(int).squeeze() - 1

            # put labels in the set of classes
            self.classes.update(list(labels))

            self.data[subfolder] = {
                "hh": hh,
                "hv": hv,
                "labels": labels,
            }

        self.classes = [int(c) for c in self.classes]

    def __len__(self):
        return sum(len(city_data["labels"]) for city_data in self.data.values())

    def find_city_and_local_idx(self, idx):
        """
        Given a global index, find the corresponding city and local index within that city's data.
        This is done in O(n_cities) time, which is acceptable since n_cities=3
        """
        cumulative = 0
        for city, city_data in self.data.items():
            city_size = len(city_data["labels"])
            if idx < cumulative + city_size:
                local_idx = idx - cumulative
                return city, local_idx
            cumulative += city_size
        raise IndexError("Index out of range")

    def __getitem__(self, idx):
        city, local_idx = self.find_city_and_local_idx(idx)

        image = np.stack(
            [
                self.data[city]["hh"][local_idx],
                self.data[city]["hv"][local_idx],
            ]
        )
        label = self.data[city]["labels"][local_idx]

        if self.transform:
            image = self.transform(image)

        return image, label
