torchcvnn.datasets
==================

.. currentmodule:: torchcvnn.datasets

Bretigny
--------

.. autoclass:: torchcvnn.datasets.Bretigny
   :members:

PolSF
-----

.. autoclass:: torchcvnn.datasets.PolSFDataset
   :members:

ALOS2
-----

We provide a generic class for parsing ALOS2 data which is the format developed
by the Japanese Aerospace Exploration Agency (JAXA).

.. autoclass:: ALOSDataset
   :members:
   :show-inheritance:

This class itself involves several parsers to process the :

- volume file :class:`torchcvnn.datasets.alos2.VolFile`
- leader file :class:`torchcvnn.datasets.alos2.LeaderFile`,
- image file :class:`torchcvnn.datasets.alos2.SARImage`,
- trailer file :class:`torchcvnn.datasets.alos2.TrailerFile`.


.. autoclass:: torchcvnn.datasets.alos2.VolFile
   :members:

.. autoclass:: torchcvnn.datasets.alos2.LeaderFile
   :members:

.. autoclass:: torchcvnn.datasets.alos2.SARImage
   :members:

.. autoclass:: torchcvnn.datasets.alos2.TrailerFile
   :members:

SLC
---

SLC is popular remote sensing format. The `Nasa Jet Lab UAV SAR <https://uavsar.jpl.nasa.gov/>`_ mission for example provides several SLC stacks.

.. autoclass:: SLCDataset
   :members:
   :show-inheritance:

This class involves several parsers for parsing :

- the annotation file :class:`torchcvnn.datasets.slc.ann_file.AnnFile`
- the SLC files :class:`torchcvnn.datasets.slc.slc_file.SLCFile`

.. autoclass:: torchcvnn.datasets.slc.ann_file.AnnFile
   :members:

.. autoclass:: torchcvnn.datasets.slc.slc_file.SLCFile
   :members:
