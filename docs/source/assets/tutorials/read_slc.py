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
import sys

# External imports
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure

# Local imports
from torchcvnn.datasets.slc.dataset import SLCDataset
from torchcvnn.datasets.slc.slc_file import SLCFile


def normalize(data):
    mod = np.abs(data)
    mod = 20 * np.log10(mod + 1e-10)

    mod = (mod - mod.min()) / (mod.max() - mod.min())  # rescale between 0 and 1
    p2, p98 = np.percentile(mod, (2, 98))

    mod = exposure.rescale_intensity(mod, in_range=(p2, p98))
    # mod = skimage.img_as_float(mod)
    return mod


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


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python read_slc.py <path_to_slc>")
        sys.exit(1)

    patch_size = (3000, 3000)
    dataset = SLCDataset(
        sys.argv[1],
        transform=get_pauli,
        patch_size=patch_size,
    )
    print(f"Dataset length : {len(dataset)}")
    patch = dataset[3]
    print(patch.shape)

    # Plot the magnitude of the patch
    plt.figure()
    plt.axis("off")
    plt.imshow(normalize(patch))
    plt.tight_layout()
    plt.savefig("patch.png", bbox_inches="tight")
    plt.show()
