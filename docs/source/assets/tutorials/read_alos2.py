# MIT License

# Copyright (c) 2023 Jérémy Fix

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

"""
Example script to read ALOS2 data

Requires additional dependencies:
        python3 -m pip install matplotlib scikit-image 
"""


# Standard imports
from pathlib import Path
import argparse
import glob
import os

# External  imports
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure

from torchcvnn.datasets import alos2


def normalize(data):
    mod = np.abs(data)
    mod = 20 * np.log10(mod + 1e-10)

    mod = (mod - mod.min()) / (mod.max() - mod.min())  # rescale between 0 and 1
    p2, p98 = np.percentile(mod, (2, 98))

    mod = exposure.rescale_intensity(mod, in_range=(p2, p98))
    # mod = skimage.img_as_float(mod)
    return mod


def get_volfile(rootdir):
    vol_search_path = os.path.join(str(rootdir), "VOL-*")
    vol_files = glob.glob(vol_search_path)
    if len(vol_files) == 0:
        raise RuntimeError(f"Cannot find any VOLUME files in {rootdir}")

    if len(vol_files) > 1:
        print(f"Warning, multiple volume files found, I will be using {vol_files[0]}")
    vol_filepath = Path(vol_files[0])
    return vol_filepath


def plot_summaries(rootdir):
    """
    Loads a ALOS-2 dataset and prints some informations extracted from the
    Volume, Leader, Trailer and Image files
    """
    # Get and find a VOL file in the provided path
    vol_filepath = get_volfile(rootdir)

    # Parse the data
    dataset = alos2.ALOSDataset(vol_filepath)

    # And print some decoded infos
    dataset.describe()


def plot_patches(rootdir):
    """
    Loads a ALOS-2 dataset and display some extracted patches
    """
    # Get and find a VOL file in the provided path
    vol_filepath = get_volfile(rootdir)

    # Limit to a subpart of the ALOS-2 data
    # This corresponds to the annotated region of the PolSF dataset
    crop_coordinates = ((2832, 736), (7888, 3520))
    dataset = alos2.ALOSDataset(
        vol_filepath,
        patch_size=(512, 512),
        patch_stride=(128, 128),
        crop_coordinates=crop_coordinates,
    )

    # Plot consecutive samples
    fig, axes = plt.subplots(1, 4, figsize=(10, 4))
    for i, ax in enumerate(axes):
        X = dataset[i]
        X = X[:, ::-1, :]
        xi = X[0]
        norm_xi = normalize(xi)
        ax.imshow(norm_xi, cmap="gray")
        ax.set_title(f"Sample {i}")
        ax.axis("off")
    plt.tight_layout()

    # Plot the four polarizations of the same patch
    X = dataset[0]
    X = X[:, ::-1, :]
    fig, axes = plt.subplots(1, 4, figsize=(10, 4))
    for xi, ax, ax_title in zip(X, axes, ["HH", "HV", "VH", "VV"]):
        norm_xi = normalize(xi)
        ax.imshow(norm_xi, cmap="gray")
        ax.axis("off")
        ax.set_title(ax_title)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "rootdir",
        type=Path,
        help="The path to a directory containing raw binary ALOS-2 data",
        nargs=1,
        default=None,
    )

    args = parser.parse_args()
    rootdir = args.rootdir[0]

    plot_summaries(rootdir)
    plot_patches(rootdir)
