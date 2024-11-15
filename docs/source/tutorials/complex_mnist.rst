Classification of MNIST digit from their Fourier representation
===============================================================

In this tutorial, we show how to create a complex valued neural network and to optimize it. As a matter of illustration,
suppose you want to classify the MNIST digits from their Fourier space representation.

Loading the complex valued MNIST dataset
----------------------------------------

Using :external:py:class:`torchvision.datasets.MNIST` and :external:py:func:`torch.fft.fft`, we can easily transform the MNIST dataset into its Fourier space representation :

.. code-block:: python

    dataset = torchvision.datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=v2_transforms.Compose([v2_transforms.PILToTensor(), torch.fft.fft]),
    )


Implementing the complex valued convolutional neural network
------------------------------------------------------------

To implement a complex valued CNN

.. code-block:: python

    cdtype = torch.complex64

    conv_model = nn.Sequential(
        *conv_block(1, 16, cdtype),
        *conv_block(16, 16, cdtype),
        *conv_block(16, 32, cdtype),
        *conv_block(32, 32, cdtype),
        nn.Flatten(),
    )

with :func:`conv_block` defined as :

.. code-block:: python


    def conv_block(in_c: int, out_c: int, cdtype: torch.dtype) -> List[nn.Module]:
        """
        Builds a basic building block of
        `Conv2d`-`Cardioid`-`Conv2d`-`Cardioid`-`AvgPool2d`

        Arguments:
            in_c : the number of input channels
            out_c : the number of output channels
            cdtype : the dtype of complex values (expected to be torch.complex64 or torch.complex32)
        """
        return [
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, dtype=cdtype),
            c_nn.BatchNorm2d(out_c),
            c_nn.Cardioid(),
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, dtype=cdtype),
            c_nn.BatchNorm2d(out_c),
            c_nn.Cardioid(),
            c_nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
        ]


And then we use a standard optimizer, training loop, cross entropy loss function, etc...

The full code is available to download :download:`mnist.py <../assets/tutorials/mnist.py>` and completly given below. To run the code, you also need the :download:`utils.py <../assets/tutorials/utils.py>` file which provides some utilitary functions. Finally, additional dependencies other than torchcvnn are needed :

.. code-block:: shell

   python3 -m pip install tochvision tqdm

If you run that script, the expected output should be :

.. code-block:: shell

   Logging to ./logs/CMNIST_0
   >> Training
   100%|██████| 844/844 [00:17<00:00, 48.61it/s]
   >> Testing
   [Step 0] Train : CE  0.20 Acc  0.94 | Valid : CE  0.08 Acc  0.97 | Test : CE 0.06 Acc  0.98[>> BETTER <<]

   >> Training
   100%|██████| 844/844 [00:16<00:00, 51.69it/s]
   >> Testing
   [Step 1] Train : CE  0.06 Acc  0.98 | Valid : CE  0.06 Acc  0.98 | Test : CE 0.05 Acc  0.98[>> BETTER <<]

   >> Training
   100%|██████| 844/844 [00:15<00:00, 53.47it/s]
   >> Testing
   [Step 2] Train : CE  0.04 Acc  0.99 | Valid : CE  0.04 Acc  0.99 | Test : CE 0.04 Acc  0.99[>> BETTER <<]

   [...]


Complete code
-------------

mnist.py
^^^^^^^^

.. literalinclude ::  ../assets/tutorials/mnist.py
   :language: python

utils.py
^^^^^^^^

.. literalinclude ::  ../assets/tutorials/utils.py
   :language: python
