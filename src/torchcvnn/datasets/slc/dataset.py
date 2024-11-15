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
import glob
import pathlib

# External imports
from torch.utils.data import Dataset
import numpy as np

# Local imports
from .slc_file import SLCFile


class SLCDataset(Dataset):
    r"""
    SLCDataset

    The format is described in https://uavsar.jpl.nasa.gov/science/documents/stack-format.html

    This object does not download the data for you, you must have the data on your local machine. For example, you can
    register and access data from the NASA JetLab https://uavsar.jpl.nasa.gov

    Note the datafiles can be quite large. For example, the quad polarization from Los Angeles SSurge_15305 is a bit
    more than 30 GB. If you take the downsampled datasets 2x8, it is 2GB.

    Note the 1x1 is 1.67 m slant range x 0.6 m azimuth.

    Note:
        As an example, using the example `read_slc.py`, with the SSurge_15305 stack provided
        by the UAVSar, the Pauli representation of the four polarizations is shown below :

        .. figure:: ../assets/datasets/slc_SSurge_15305.png
           :alt: Pauli representation of a :math:`3000 \times 3000` crop of the SSurge_15305 stack
           :width: 50%
           :align: center


        The code may look like this :

        .. code-block:: python

            import numpy as np
            import torchcvnn
            from torchcvnn.datasets.slc.dataset import SLCDataset

            def get_pauli(data):
                # Returns Pauli in (H, W, C)
                HH = data["HH"]
                HV = data["HV"]
                VH = data["VH"]
                VV = data["VV"]

                alpha = HH + VV
                beta = HH - VV
                gamma = HV + VH

                return np.stack([beta, gamma, alpha], axis=-1)


            patch_size = (3000, 3000)
            dataset = SLCDataset(
                rootdir,
                transform=get_pauli,
                patch_size=patch_size,
            )

    Arguments:
        rootdir: the path containing the SLC and ANN files
        transform : the transform applied to the patches. It applies
                    on a dictionnary of patches {'HH': np.array, 'HV': np.array, ...}
        patch_size: the dimensions of the patches to consider (rows, cols)
        patch_stride: the shift between two consecutive patches, default:patch_size
    """

    def __init__(
        self,
        rootdir: str = None,
        transform=None,
        patch_size: tuple = (128, 128),
        patch_stride: tuple = None,
    ):
        super().__init__()

        self.transform = transform
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        if self.patch_stride is None:
            self.patch_stride = patch_size

        # Let us find all the SLC files
        # We group the polorizations of the stack together
        self.slcs = glob.glob(str(pathlib.Path(rootdir) / "*.slc"))
        self.slc_polarizations = {}
        self.patch_counts = {}
        for slc in self.slcs:
            slc_file = SLCFile(slc, patch_size=patch_size, patch_stride=patch_stride)
            slc_key = slc_file.key
            if slc_key not in self.slc_polarizations:
                self.slc_polarizations[slc_key] = {}

            self.slc_polarizations[slc_key][slc_file.polarization] = slc_file
            self.patch_counts[slc_key] = len(slc_file)

        # Sanity checks
        # 1- For every SLC stack, we must have the same number of patches
        # 2- All the SLCs must have the same number of polarizations
        polarization_count = None
        for slc_key, slc_polarizations in self.slc_polarizations.items():
            if polarization_count is None:
                polarization_count = len(slc_polarizations)
            else:
                assert polarization_count == len(slc_polarizations)

            patch_count = None
            for polarization, slc_file in slc_polarizations.items():
                if patch_count is None:
                    patch_count = len(slc_file)
                else:
                    assert patch_count == len(slc_file)

        self.nsamples = sum(self.patch_counts.values())

    def __len__(self):
        return self.nsamples

    def __getitem__(self, item):
        """
        Usefull params :
        - 1x1_slc_azimuth_pixel_spacing
        - 1x1_slc_range_pixel_spacing
        - global_average_squint_angle
        - center_wavelength
        - slc_SEGMENT_1x1_rows
        - slc_SEGMENT_1x1_columns
        """
        # Now, request the patch from the right SLC file
        assert 0 <= item < self.nsamples

        # 1- Find the right SLC file given self.patch_counts and item index
        for slc_key, count in self.patch_counts.items():
            if item < count:
                slcs = self.slc_polarizations[slc_key]
                break
            else:
                item -= count
        sorted_keys = sorted(slcs.keys())
        # 2- Find the right patch from all the polarizations
        patches = {pol: slcs[pol][item] for pol in sorted_keys}

        # 3a- Stack the patches
        # 3b- Apply the transform
        if self.transform is not None:
            patches = self.transform(patches)
        else:
            patches = np.stack([patchi for _, patchi in patches.items()])

        return patches
