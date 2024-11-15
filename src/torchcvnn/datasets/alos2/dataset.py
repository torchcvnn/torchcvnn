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
# External imports
from torch.utils.data import Dataset
import numpy as np

# Local imports
from . import VolFile, LeaderFile, TrailerFile, SARImage


class ALOSDataset(Dataset):
    r"""
    ALOSDataset

    The format is described in
    `<https://www.eorc.jaxa.jp/ALOS/en/alos-2/pdf/product_format_description/PALSAR-2_xx_Format_CEOS_E_g.pdf>`_

    The dataset is constructed from the volume file. If leader and trailer files
    are colocated, they are loaded as well.

    Important, this code has been developed for working with L1.1 HBQ-R Quad Pol
    datafiles. It is not expected to work out of the box for other levels and
    for less than 4 polarizations.

    Arguments:
        volpath: the path to the VOLUME file
        transform : the transform applied the cropped image. It applies
                    on a dictionnary of patches {'HH': np.array, 'HV': np.array}
        crop_coordinates: the subpart of the image to consider as ((row_i, col_i), (row_j, col_j))
                          defining the corner coordinates
        patch_size: the dimensions of the patches to consider (rows, cols)
        patch_stride: the shift between two consecutive patches, default:patch_size
    """

    def __init__(
        self,
        volpath: str = None,
        transform=None,
        crop_coordinates: tuple = None,
        patch_size: tuple = (128, 128),
        patch_stride: tuple = None,
    ):
        super().__init__()

        self.transform = transform

        self.patch_size = patch_size
        self.patch_stride = patch_stride
        if patch_stride is None:
            self.patch_stride = patch_size

        self.volFile = VolFile(volpath)

        leader_filepath = volpath.parents[0] / volpath.name.replace("VOL-", "LED-")
        self.leaderFile = None
        if leader_filepath.exists():
            self.leaderFile = LeaderFile(leader_filepath)

        trailer_filepath = volpath.parents[0] / volpath.name.replace("VOL-", "TRL-")
        self.trailerFile = None
        if trailer_filepath.exists():
            self.trailerFile = TrailerFile(trailer_filepath)

        self.crop_coordinates = None
        if crop_coordinates is not None:
            self.crop_coordinates = crop_coordinates

        self.images = {}
        for pol in ["HH", "HV", "VH", "VV"]:
            filepath = volpath.parents[0] / volpath.name.replace("VOL-", f"IMG-{pol}-")
            if not filepath.exists():
                continue
            self.images[pol] = SARImage(filepath)
            if self.crop_coordinates is None:
                self.crop_coordinates = (
                    (0, 0),
                    (self.images[pol].num_rows, self.images[pol].num_cols),
                )

        if len(self.images) != self.volFile.num_polarizations:
            raise RuntimeError(
                f"I was expecting {self.volFile.num_polarizations} data file but I found {len(self.images)} data file"
            )

        # Precompute the dimension of the grid of patches
        nrows = self.crop_coordinates[1][0] - self.crop_coordinates[0][0]
        ncols = self.crop_coordinates[1][1] - self.crop_coordinates[0][1]

        nrows_patch, ncols_patch = self.patch_size
        row_stride, col_stride = self.patch_stride

        self.nsamples_per_rows = (nrows - nrows_patch) // row_stride + 1
        self.nsamples_per_cols = (ncols - ncols_patch) // col_stride + 1

    @property
    def polarizations(self):
        return self.images.keys()

    def describe(self):
        print(
            f"""
Volume File
===========
{self.volFile}

Leader File
===========
{self.leaderFile}

Trailer File
===========
{self.trailerFile}
"""
        )

    def __len__(self) -> int:
        """
        Returns the length of the dataset according to the patch size, stride
        and image size

        Returns:
            int: the total number of available patches
        """

        return self.nsamples_per_rows * self.nsamples_per_cols

    def __getitem__(self, idx: int):
        """
        Access and returns the subpatch specified by the index

        Arguments:
            idx: the index of the patch to access
        """
        row_stride, col_stride = self.patch_stride
        start_row = (
            self.crop_coordinates[0][0] + (idx // self.nsamples_per_cols) * row_stride
        )
        start_col = (
            self.crop_coordinates[0][1] + (idx % self.nsamples_per_cols) * col_stride
        )
        num_rows, num_cols = self.patch_size
        patches = {
            pol: im.read_patch(start_row, num_rows, start_col, num_cols)
            * self.leaderFile.calibration_factor
            for pol, im in self.images.items()
        }

        if self.transform is not None:
            patches = self.transform(patches)
        else:
            patches = np.stack([patchi for _, patchi in patches.items()])

        return patches
