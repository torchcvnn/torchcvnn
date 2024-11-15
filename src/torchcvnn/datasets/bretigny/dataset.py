# MIT License

# Copyright (c) 2024 Chengfang Ren, Jeremy Fix

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
from torch.utils.data import Dataset
import numpy as np


class Bretigny(Dataset):
    r"""
    Bretigny Dataset

    Arguments:
        root: the root directory containing the npz files for Bretigny
        fold: train (70%), valid (15%), or test (15%)
        transform : the transform applied the cropped image
        balanced: whether or not to use balanced labels
        patch_size: the dimensions of the patches to consider (rows, cols)
        patch_stride: the shift between two consecutive patches, default:patch_size

    Note:
        An example usage :

        .. code-block:: python

            import torchcvnn
            from torchcvnn.datasets import Bretigny

            dataset = Bretigny(
                rootdir, fold="train", patch_size=((128, 128)), transform=lambda x: np.abs(x)
            )
            X, y = dataset[0]

        Displayed below are the train, valid and test parts with the labels overlayed

        .. image:: ../assets/datasets/bretigny_train.png
           :alt: Train fold
           :width: 60%
        .. image:: ../assets/datasets/bretigny_valid.png
           :alt: Valid fold
           :width: 15%
        .. image:: ../assets/datasets/bretigny_test.png
           :alt: Test fold
           :width: 15%

    """

    """
    Class names
    """
    classes = ["0 - Unlabeld", "1 - Forest", "2 - Track", "3 - Urban", "4 - Fields"]

    def __init__(
        self,
        root: str,
        fold: str,
        transform=None,
        balanced: bool = False,
        patch_size: tuple = (128, 128),
        patch_stride: tuple = None,
    ):
        self.root = pathlib.Path(root)
        self.transform = transform

        self.patch_size = patch_size
        self.patch_stride = patch_stride
        if patch_stride is None:
            self.patch_stride = patch_size

        # Preload the data
        sar_filename = self.root / "bretigny_seg.npz"
        if not sar_filename.exists():
            raise RuntimeError(f"Cannot find the file {sar_filename}")
        sar_data = np.load(sar_filename)
        self.HH, self.HV, self.VV = sar_data["HH"], sar_data["HV"], sar_data["VV"]

        if balanced:
            label_filename = self.root / "bretigny_seg_4ROI_balanced.npz"
        else:
            label_filename = self.root / "bretigny_seg_4ROI.npz"
        if not label_filename.exists():
            raise RuntimeError(f"Cannot find the label file {label_filename}")
        self.labels = np.load(label_filename)["arr_0"]

        if not fold in ["train", "valid", "test"]:
            raise ValueError(
                f"Unrecognized fold {fold}. Should be either train, valid or test"
            )

        # Crop the data with respect to the fold
        if fold == "train":
            col_start = 0
            col_end = int(0.70 * self.HH.shape[1])
        elif fold == "valid":
            col_start = int(0.70 * self.HH.shape[1]) + 1
            col_end = int(0.85 * self.HH.shape[1])
        else:
            col_start = int(0.85 * self.HH.shape[1]) + 1
            col_end = self.HH.shape[1]

        self.HH = self.HH[:, col_start:col_end]
        self.HV = self.HV[:, col_start:col_end]
        self.VV = self.VV[:, col_start:col_end]
        self.labels = self.labels[:, col_start:col_end]

        # Precompute the dimension of the grid of patches
        nrows = self.HH.shape[0]
        ncols = self.HH.shape[1]

        nrows_patch, ncols_patch = self.patch_size
        row_stride, col_stride = self.patch_stride

        self.nsamples_per_rows = (nrows - nrows_patch) // row_stride + 1
        self.nsamples_per_cols = (ncols - ncols_patch) // col_stride + 1

    def __len__(self) -> int:
        """
        Returns the total number of patches in the whole image.

        Returns:
            the total number of patches in the dataset
        """
        return self.nsamples_per_rows * self.nsamples_per_cols

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        """
        Returns the indexes patch.

        Arguments:
            idx (int): Index

        Returns:
            tuple: (patch, labels) where patch contains the 3 complex valued polarization HH, HV, VV and labels contains the aligned semantic labels
        """
        row_stride, col_stride = self.patch_stride
        start_row = (idx // self.nsamples_per_cols) * row_stride
        start_col = (idx % self.nsamples_per_cols) * col_stride
        num_rows, num_cols = self.patch_size
        patches = [
            patch[
                start_row : (start_row + num_rows), start_col : (start_col + num_cols)
            ]
            for patch in [self.HH, self.HV, self.VV]
        ]
        patches = np.stack(patches)
        if self.transform is not None:
            patches = self.transform(patches)

        labels = self.labels[
            start_row : (start_row + num_rows), start_col : (start_col + num_cols)
        ]

        return patches, labels
