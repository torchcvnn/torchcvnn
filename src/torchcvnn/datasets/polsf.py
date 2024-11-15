# MIT License

# Copyright (c) 2024 Quentin Gabot

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
import pathlib
from typing import Tuple, Any

# External imports
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

# Local imports
from .alos2 import ALOSDataset


class PolSFDataset(Dataset):
    r"""
    The Polarimetric SAR dataset with the labels provided by
    https://ietr-lab.univ-rennes1.fr/polsarpro-bio/san-francisco/

    We expect the data to be already downloaded and available on your drive.

    Arguments:
        root: the top root dir where the data are expected
        transform : the transform applied the cropped image
        patch_size: the dimensions of the patches to consider (rows, cols)
        patch_stride: the shift between two consecutive patches, default:patch_size

    Note:
        An example usage :

        .. code-block:: python

            import torchcvnn
            from torchcvnn.datasets import PolSFDataset

            def transform_patches(patches):
                # We keep all the patches and get the spectrum
                # from it
                # If you wish, you could filter out some polarizations
                # PolSF provides the four HH, HV, VH, VV
                patches = [np.abs(patchi) for _, patchi in patches.items()]
                return np.stack(patches)

            dataset = PolSFDataset(rootdir, patch_size=((512, 512)), transform=transform_patches
            X, y = dataset[0]

        Displayed below are example patches with patch sizes :math:`512 \times 512`
        with the labels overlayed

        .. figure:: ../assets/datasets/polsf.png
           :alt: Patches from the PolSF dataset
           :width: 100%
           :align: center

    """

    """
    Class names
    """
    classes = [
        "0 - unlabel",
        "1 - Montain",
        "2 - Water",
        "3 - Vegetation",
        "4 - High-Density Urban",
        "5 - Low-Density Urban",
        "6 - Developd",
    ]

    def __init__(
        self,
        root: str,
        transform=None,
        patch_size: tuple = (128, 128),
        patch_stride: tuple = None,
    ):
        self.root = root

        # alos2_url = "https://ietr-lab.univ-rennes1.fr/polsarpro-bio/san-francisco/dataset/SAN_FRANCISCO_ALOS2.zip"
        # labels_url = "https://raw.githubusercontent.com/liuxuvip/PolSF/master/SF-ALOS2/SF-ALOS2-label2d.png"

        crop_coordinates = ((2832, 736), (7888, 3520))
        root = pathlib.Path(root) / "VOL-ALOS2044980750-150324-HBQR1.1__A"
        self.alos_dataset = ALOSDataset(
            root, transform, crop_coordinates, patch_size, patch_stride
        )
        if isinstance(root, str):
            root = pathlib.Path(root)
        self.labels = np.array(Image.open(root.parent / "SF-ALOS2-label2d.png"))[
            ::-1, :
        ].copy()  # copy necessary as otherwise torch.from_numpy does not support
        # negative stride

    def __len__(self) -> int:
        """
        Returns the total number of patches in the while image.

        Returns:
            the total number of patches in the dataset
        """
        return len(self.alos_dataset)

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        """
        Returns the indexes patch.

        Arguments:
            idx (int): Index

        Returns:
            tuple: (patch, labels) where patch contains the 4 complex valued polarization HH, HV, VH, VV and labels contains the aligned semantic labels
        """
        alos_patch = self.alos_dataset[idx]

        row_stride, col_stride = self.alos_dataset.patch_stride
        start_row = (idx // self.alos_dataset.nsamples_per_cols) * row_stride

        start_col = (idx % self.alos_dataset.nsamples_per_cols) * col_stride
        num_rows, num_cols = self.alos_dataset.patch_size
        labels = self.labels[
            start_row : (start_row + num_rows), start_col : (start_col + num_cols)
        ]

        return alos_patch, labels
