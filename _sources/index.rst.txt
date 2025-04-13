Complex Valued Neural Networks
==============================

This is a library that uses `pytorch <https://pytorch.org>`_ as a back-end for complex valued neural networks.

It was initially developed by Victor Dhédin and Jérémie Levi during their third year project at `CentraleSupélec <https://www.centralesupelec.fr/>`_. 

You can access it on github.

You can access the code coverage report `here <https://torchcvnn.github.io/torchcvnn/htmlcov>`_.

.. raw:: html

    <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16" width="30" height="30">
    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
    </svg><a href="https://github.com/torchcvnn/torchcvnn"> https://github.com/torchcvnn/torchcvnn</a>

Other related projects
----------------------

You might also be interested in some other projects: 

Tensorflow based : 

- `cvnn <https://github.com/NEGU93/cvnn>`_ developed by colleagues from CentraleSupélec

Pytorch based : 

- `cplxmodule <https://github.com/ivannz/cplxmodule>`_
- `complexPyTorch <https://github.com/wavefrontshaping/complexPyTorch>`_


.. toctree::
   :maxdepth: 2
   :caption: Getting started
   :hidden:

   getting_started/installing


.. toctree::
   :maxdepth: 2
   :caption: Package Reference
   :hidden:

   modules/datasets
   modules/transforms
   modules/nn
   modules/models
   modules/init

.. toctree::
   :maxdepth: 2
   :caption: Tutorials
   :hidden:

   tutorials/complex_mnist
   tutorials/loading_remote
