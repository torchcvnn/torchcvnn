# Complex-Valued Neural Networks (CVNN) - Pytorch

[![docs](https://github.com/torchcvnn/torchcvnn/actions/workflows/doc.yml/badge.svg)](https://torchcvnn.github.io/torchcvnn/) ![pytest](https://github.com/torchcvnn/torchcvnn/actions/workflows/test.yml/badge.svg) [![PyPI version](https://badge.fury.io/py/torchcvnn.svg)](https://badge.fury.io/py/torchcvnn)

**Documentation** [https://torchcvnn.github.io/torchcvnn/](https://torchcvnn.github.io/torchcvnn/)

**Examples** [https://www.github.com/torchcvnn/examples](https://www.github.com/torchcvnn/examples)

This is a library that uses [pytorch](https://pytorch.org) as a back-end for complex valued neural networks. It provides :

- complex valued datasets from remote sensing and MRI,
- complex valued transforms,
- complex valued layers, some of them requiring specific implementations because their computation is specific with complex valued activations, others are implemented merely because the lower level implementation raise an exception if processing complex valued activations even though the computations are the same than for real valued activations
- complex valued neural networks

It was initially developed by Victor Dhédin and Jérémie Levi during their third year project at CentraleSupélec. 

## Installation

To install the library, it is simple as :

```
python -m pip install torchcvnn
```

or, using [uv](https://docs.astral.sh/uv/) :

```
uv pip install torchcvnn
```


## Other projects

You might also be interested in some other projects: 

Tensorflow based : 

- [cvnn](https://github.com/NEGU93/cvnn) developed by colleagues from CentraleSupélec

Pytorch based : 

- [cplxmodule](https://github.com/ivannz/cplxmodule)
- [complexPyTorch](https://github.com/wavefrontshaping/complexPyTorch)
- [complextorch](https://github.com/josiahwsmith10/complextorch)
