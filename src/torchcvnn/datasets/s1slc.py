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

    def __init__(self, root, transform=None):
        self.transform = transform
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
            hh = np.load(hh_path)
            hv = np.load(hv_path)
            label = np.load(labels_path)
            label = [int(l.item()) - 1 for l in label]  # Convert to 0-indexed labels

            # Concatenate HH and HV to create a two-channel array
            data = np.stack((hh, hv), axis=1)  # Shape: (B, 2, H, W)

            # Append data and labels to the lists
            self.data.extend(data)
            self.labels.extend(label)

        self.classes = list(set(self.labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        return image, label
