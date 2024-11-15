Remote sensing data loading
===========================

This example illustres some data loading utilities provided by torchcvnn for remote sensing.

SLC
---

We provide the generic :py:class:`torchcvnn.datasets.SLCDataset` object as a generic SLC data parser. An example script
using it can be found in `read_slc.py <../assets/tutorials/read_slc.py>`_. This object can be used to load all the SLC
files present in a directory. The parser requires both the SLC and annotation files to be present in the same directory. 

A basic example of how to use the SLC parser is shown below:

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

ALOS2
-----

We also provide a generic parser for the JAXA ALOS2 format as the generic :py:class:`torchcvnn.datasets.ALOSDataset`. An example script
using it can be found in `read_alos2.py <../assets/tutorials/read_alos2.py>`_. If the trailer and leader files are
colocated with the ALOS2 volume file, they are loaded as well.

The ALOS parser allows for cropping a subpart of the data. The example below shows how to do that :

.. code-block:: python

   crop_coordinates = ((2832, 736), (7888, 3520))
   dataset = alos2.ALOSDataset(
       vol_filepath,
       patch_size=(512, 512),
       patch_stride=(128, 128),
       crop_coordinates=crop_coordinates,
   )

