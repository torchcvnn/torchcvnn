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
from torch.utils.data import Dataset
import numpy as np
import torch


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

    def __init__(self, root, transform=None, lazy_loading=True):
        self.transform = transform
        self.lazy_loading = lazy_loading
        # Get list of subfolders in the root path
        subfolders = [
            os.path.join(root, name)
            for name in os.listdir(root)
            if os.path.isdir(os.path.join(root, name))
        ]

        self.data = []
        self.labels = []

        for subfolder in subfolders:
            # Define paths to the .npy files
            hh_path = os.path.join(subfolder, "HH.npy")
            hv_path = os.path.join(subfolder, "HV.npy")
            labels_path = os.path.join(subfolder, "Labels.npy")

            # Load the .npy files
            hh = np.load(hh_path, mmap_mode="r")
            hv = np.load(hv_path, mmap_mode="r")

            if not lazy_loading:
                # If not lazy loading, we load all the data in main memory
                # Concatenate HH and HV to create a two-channel array
                data = np.stack((hh, hv), axis=1)  # Shape: (B, 2, H, W)
            else:
                # If lazy loading, we store the paths to the .npy files
                num_patches = hh.shape[0]
                data = [
                    (hh_path, hv_path, patch_idx) for patch_idx in range(num_patches)
                ]

            # For the labels, we can preload everything in main memory
            label = np.load(labels_path, mmap_mode="r")
            label = [int(l.item()) - 1 for l in label]  # Convert to 0-indexed labels

            # Append data and labels to the lists
            self.data.extend(data)
            self.labels.extend(label)

        self.classes = list(set(self.labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        if self.lazy_loading:
            hh_path, hv_path, patch_idx = self.data[idx]

            # Load the .npy files
            hh = np.load(hh_path)
            hv = np.load(hv_path)

            # Extract the right patch
            hh_patch = hh[patch_idx]
            hv_patch = hv[patch_idx]

            # Concatenate HH and HV to create a two-channel array
            image = np.stack((hh_patch, hv_patch), axis=0)  # Shape: (2, H, W)
        else:
            image = self.data[idx]

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
