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


Normalization layers
--------------------

.. autosummary::
   :toctree: _autosummary

   BatchNorm2d
   BatchNorm1d
   LayerNorm

Dropout layers
--------------
