torchcvnn.nn
============

.. currentmodule:: torchcvnn.nn

Convolution layers
------------------

.. autosummary::
   :toctree: _autosummary

   ConvTranspose2d

Pooling layers
--------------

.. autosummary::
   :toctree: _autosummary

   MaxPool2d
   AvgPool2d
   AdaptiveAvgPool2d


UpSampling layers
-----------------

.. autosummary::
   :toctree: _autosummary

   Upsample

Activations
-----------

Type A
^^^^^^

These activation functions apply the same function independently to both real and imaginary components.

.. autosummary::
   :toctree: _autosummary

   CCELU
   CELU
   CGELU
   CReLU
   CPReLU
   CSigmoid
   CTanh



Type B
^^^^^^

The Type B activation functions take into account both the magnitude and phase of the input.

.. autosummary::
   :toctree: _autosummary

    Cardioid
    Mod
    modReLU
    zAbsReLU
    zLeakyReLU
    zReLU


Attention
^^^^^^^^^

.. autosummary::
   :toctree: _autosummary

   MultiheadAttention

Normalization layers
--------------------

.. autosummary::
   :toctree: _autosummary

   BatchNorm2d
   BatchNorm1d
   LayerNorm
   RMSNorm

Transformer layers
------------------

.. autosummary::
   :toctree: _autosummary

   Transformer
   TransformerEncoderLayer
   TransformerDecoderLayer

Vision Transformer
------------------

.. autosummary::
   :toctree: _autosummary

   ViTLayer
   ViT

Dropout layers
--------------


.. autosummary::
   :toctree: _autosummary

   Dropout
   Dropout2d
