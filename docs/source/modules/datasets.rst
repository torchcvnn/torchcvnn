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

S1SLC
-----

S1SLC is a dataset built from Sentinel-1 SLC data. The dataset is provided at the following url `https://ieee-dataport.org/open-access/s1slccvdl-complex-valued-annotated-single-look-complex-sentinel-1-sar-dataset-complex`_.

.. autoclass:: S1SLC
   :members:
   :show-inheritance:

MSTAR
-----

MSTAR is a popular radar dataset where the task is to classify military vehicles (tanks, trucks, guns, bulldozer, etc).
To use this dataset, you need to manually download the data before hand and to unpack them into the same directory :


- MSTAR_PUBLIC_T_72_VARIANTS_CD1 : https://www.sdms.afrl.af.mil/index.php?collection=mstar&page=variants
- MSTAR_PUBLIC_MIXED_TARGETS_CD1 : https://www.sdms.afrl.af.mil/index.php?collection=mstar&page=mixed
- MSTAR_PUBLIC_MIXED_TARGETS_CD2 : https://www.sdms.afrl.af.mil/index.php?collection=mstar&page=mixed
- MSTAR_PUBLIC_TARGETS_CHIPS_T72_BMP2_BTR70_SLICY : https://www.sdms.afrl.af.mil/index.php?collection=mstar&page=targets

.. autoclass:: MSTARTargets
   :members:

SAMPLE
------

SAMPLE is a dataset built from real SAR data as provided by the MSTAR dataset as well a synthetic data. As the original
MSTAR dataset, it contains military vehicles and actually a subset of 10 classes : 2s1, bmp2, btr70, m1, m2, m35, m548, m60, t72, zsu23 . It contains a total of 3968 samples. The SAMPLE dataset is provided by https://github.com/benjaminlewis-afrl/SAMPLE_dataset_public .

.. autoclass:: SAMPLE
   :members:


MICCAI2023
----------

The MICCAI2023 challenge involved the task of cine reconstruction where the objective is to predict a full sampled
k-space from an under-sampled k-space. The data come from cardiac MRI.

.. autofunction:: torchcvnn.datasets.miccai2023.kspace_to_image

.. autofunction:: torchcvnn.datasets.miccai2023.image_to_kspace

.. autofunction:: torchcvnn.datasets.miccai2023.combine_coils_from_kspace

.. autoclass:: MICCAI2023
   :members:


ATRNet-STAR
-----------

ATRNet-STAR is a large vehicule classification/detection dataset of SAR images provided by `Liu et al.(2025) <https://arxiv.org/abs/2501.13354>`_ and released on huggingface hub (`https://huggingface.co/datasets/waterdisappear/ATRNet-STAR`_). It contains :

- 40 target types with 17003 training samples and 7243 test samples.
- quad polarization
- bands X, Ku

.. autoclass:: ATRNetSTAR
   :members: