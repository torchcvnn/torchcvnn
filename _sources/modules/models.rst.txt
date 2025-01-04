torchcvnn.models
================

.. currentmodule:: torchcvnn.models

Vision Transformers
-------------------

We provide some predefined ViT models. Their configurations are listed below.

.. list-table:: ViT tiny model configuration
    :header-rows: 1

    * - Model name
      - Layers
      - Heads
      - Hidden dimension
      - MLP dimension
      - Dropout
      - Attention dropout
      - Norm layer
    * - vit_t
      - 12
      - 3
      - 192
      - 768
      - 0.0
      - 0.0
      - RMSNorm
    * - vit_s
      - 12
      - 6
      - 384
      - 1536
      - 0.0
      - 0.0
      - RMSNorm
    * - vit_b
      - 12
      - 12
      - 768
      - 3072
      - 0.0
      - 0.0
      - RMSNorm
    * - vit_l
      - 24
      - 16
      - 1024
      - 4096
      - 0.0
      - 0.0
      - RMSNorm
    * - vit_h
      - 32
      - 16
      - 1280
      - 5120
      - 0.0
      - 0.0
      - RMSNorm


.. autofunction:: torchcvnn.models.vit_t

.. autofunction:: torchcvnn.models.vit_s

.. autofunction:: torchcvnn.models.vit_b

.. autofunction:: torchcvnn.models.vit_l

.. autofunction:: torchcvnn.models.vit_h
