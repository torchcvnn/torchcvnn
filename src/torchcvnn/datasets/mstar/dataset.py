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
from typing import Tuple
import pathlib
import logging
import struct

# External imports
import torch
from torch.utils.data import Dataset
import numpy as np


def parse_header(fh) -> dict:
    """
    This function parses the PhoenixHeader from a file handle from the provided file handle.

    It returns a dictionnary containing all the fields of the PhoenixHeader.

    It raises an exception if the header is not valid, i.e. does not start with [PhoenixHeaderVer01.04] or [PhoenixHeaderVer01.05]
    """
    parsed_fields = {}
    # There is one character at the very beginning of the file, before the header
    _ = fh.readline()
    start_line = fh.readline().strip()
    accepted_headers = ["[PhoenixHeaderVer01.04]", "[PhoenixHeaderVer01.05]"]
    if start_line not in accepted_headers:
        raise ValueError(
            f"Invalid header : {start_line}, expected one of {accepted_headers}"
        )
    next_line = fh.readline().strip()
    while next_line != "[EndofPhoenixHeader]":
        items = next_line.split("=")
        key = items[0].strip()
        value = items[1].strip()
        parsed_fields[key] = value
        next_line = fh.readline().strip()

    return parsed_fields


class MSTARSample:
    """
    This class implements a sample from the MSTAR dataset.

    The extracted complex image is stored in the data attribute.
    The data is a numpy array of shape (num_rows, num_cols, 1)

    The header is stored in the header attribute. It contains all the fields
    of the PhoenixHeader.

    Arguments:
        filename : the name of the file to load
    """

    def __init__(self, filename: str):
        self.filename = pathlib.Path(filename)

        with open(self.filename, "r", errors="replace") as fh:
            # Read the header from the file
            self.header = parse_header(fh)

            num_rows = int(self.header["NumberOfRows"])
            num_cols = int(self.header["NumberOfColumns"])
            phoenix_header_length = int(self.header["PhoenixHeaderLength"])
            native_header_length = int(self.header["native_header_length"])

            fh.seek(phoenix_header_length + native_header_length)

            sig_size = int(self.header["PhoenixSigSize"])
            bytes_per_values = (
                sig_size - phoenix_header_length - native_header_length
            ) // (2 * num_rows * num_cols)

            if bytes_per_values == 4:
                # Read the data as float32
                pass
            elif bytes_per_values == 2:
                # Read the data as uint16
                pass
            else:
                raise ValueError(
                    f"Unsupported number of bytes per value : {bytes_per_values}"
                )

        # Read the data from the file
        with open(self.filename, "rb") as fh:
            fh.seek(phoenix_header_length + native_header_length)

            data_bytes = fh.read(num_rows * num_cols * bytes_per_values)
            unpacked = struct.unpack(">" + ("f" * num_cols * num_rows), data_bytes)
            magnitudes = np.array(unpacked).reshape(num_rows, num_cols)

            data_bytes = fh.read(num_rows * num_cols * bytes_per_values)
            unpacked = struct.unpack(">" + ("f" * num_cols * num_rows), data_bytes)
            phases = np.array(unpacked).reshape(num_rows, num_cols)

            self.data = magnitudes * np.exp(1j * phases)
        self.data = self.data[:, :, np.newaxis]


def gather_mstar_datafiles(rootdir: pathlib.Path, target_name_depth: int = 1) -> dict:
    """
    This function gathers all the MSTAR datafiles from the root directory
    It looks for files named HBxxxx that are data files (containing a PhoenixHeader).

    The assigned target name is the name of the directory, or parent directory, at the target_name_depth level.
    """

    data_files = {}
    for filename in rootdir.glob("**/HB*"):
        if not filename.is_file():
            continue

        try:
            with open(filename, "r", errors="replace") as fh:
                _ = parse_header(fh)
            # sample = MSTARSample(filename)
        except Exception as e:
            logging.debug(
                f"The file {filename} failed to be loaded as a MSTAR sample: {e}"
            )
            continue

        target_name = filename.parts[-target_name_depth]
        if target_name not in data_files:
            data_files[target_name] = []

        logging.debug(f"Successfully parsed {filename} as a {target_name} sample.")
        data_files[target_name].append(filename)
    return data_files


class MSTARTargets(Dataset):
    """
    This class implements a PyTorch Dataset for the MSTAR dataset.

    The MSTAR dataset is composed of several sub-datasets. The datasets must
    be downloaded manually because they require authentication.

    To download these datasets, you must register at the following address: https://www.sdms.afrl.af.mil/index.php?collection=mstar

    This dataset object expects all the datasets to be unpacked in the same directory. We can parse the following :

    - MSTAR_PUBLIC_T_72_VARIANTS_CD1 : https://www.sdms.afrl.af.mil/index.php?collection=mstar&page=variants
    - MSTAR_PUBLIC_T_72_VARIANTS_CD2 : https://www.sdms.afrl.af.mil/index.php?collection=mstar&page=variants
    - MSTAR_PUBLIC_MIXED_TARGETS_CD1 : https://www.sdms.afrl.af.mil/index.php?collection=mstar&page=mixed
    - MSTAR_PUBLIC_MIXED_TARGETS_CD2 : https://www.sdms.afrl.af.mil/index.php?collection=mstar&page=mixed
    - MSTAR_PUBLIC_TARGETS_CHIPS_T72_BMP2_BTR70_SLICY :
      https://www.sdms.afrl.af.mil/index.php?collection=mstar&page=targets

    Arguments:
        rootdir : str
        transform : the transform applied on the input complex valued array

    Note:
        An example usage :

        .. code-block:: python

            import torchcvnn
            from torchcvnn.datasets import MSTARTargets

            transform = v2.Compose(
                transforms=[v2.ToImage(), v2.Resize(128), v2.CenterCrop(128)]
            )
            dataset = MSTARTargets(
                rootdir, transform=transform
            )
            X, y = dataset[0]

        Displayed below are some examples for every class in the dataset. To plot them, we extracted
        only the magnitude of the signals although the data are indeed complex valued.

        .. image:: ../assets/datasets/mstar.png
           :alt: Samples from MSTAR
           :width: 60%


    """

    def __init__(self, rootdir: str, transform=None):
        super().__init__()
        self.rootdir = pathlib.Path(rootdir)
        self.transform = transform

        rootdir = pathlib.Path(rootdir)

        # The MSTAR dataset is composed of several sub-datasets
        # Each sub-dataset has a different layout
        # The dictionnary below maps the directory name of the sub-dataset
        # to the depth at which the target name is located in the directory structure
        # with respect to a datafile
        sub_datasets = {
            "MSTAR_PUBLIC_T_72_VARIANTS_CD1": 2,
            "MSTAR_PUBLIC_T_72_VARIANTS_CD2": 2,
            "MSTAR_PUBLIC_MIXED_TARGETS_CD1": 2,
            "MSTAR_PUBLIC_MIXED_TARGETS_CD2": 2,
            "MSTAR_PUBLIC_TARGETS_CHIPS_T72_BMP2_BTR70_SLICY": 3,
        }

        # We collect all the samples from all the sub-datasets
        self.data_files = {}
        for sub_dataset, target_name_depth in sub_datasets.items():
            sub_dir = rootdir / sub_dataset
            if not sub_dir.exists():
                logging.warning(f"Directory {sub_dir} does not exist.")
                continue
            # Append the data files from the sub-dataset
            for key, value in gather_mstar_datafiles(
                sub_dir, target_name_depth
            ).items():
                if key not in self.data_files:
                    self.data_files[key] = []
                self.data_files[key].extend(value)
        self.class_names = list(self.data_files.keys())

        # We then count how many samples have been loaded for all the classes
        self.num_data_files = {}
        self.tot_num_data_files = 0
        for key in self.class_names:
            self.num_data_files[key] = len(self.data_files[key])
            self.tot_num_data_files += self.num_data_files[key]

        logging.debug(
            f"Loaded {self.tot_num_data_files} MSTAR samples from the following classes : {self.class_names}."
        )
        # List the number of samples per class
        for key in self.class_names:
            logging.debug(f"Class {key} : {self.num_data_files[key]} samples.")

    def __len__(self) -> int:
        """
        Returns the total number of samples
        """
        return self.tot_num_data_files

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Returns the sample at the given index. Applies the transform
        if provided. The type of the first component of the tuple
        depends on the provided transform. If None is provided, the
        sample is a complex valued numpy array.

        Arguments:
            index : index of the sample to return

        Returns:
            data : the sample
            class_idx : the index of the class in the class_names list
        """

        if index >= self.tot_num_data_files:
            raise IndexError

        # We look for the class from which the sample will be taken
        for key in self.data_files.keys():
            if index < self.num_data_files[key]:
                break
            index -= self.num_data_files[key]

        filename = self.data_files[key][index]
        logging.debug(f"Loading the MSTAR file {filename}")

        sample = MSTARSample(filename)
        class_idx = self.class_names.index(key)

        data = sample.data
        if self.transform is not None:
            data = self.transform(data)

        return data, class_idx
