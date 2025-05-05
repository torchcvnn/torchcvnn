# Complex-Valued Neural Networks (CVNN) - Pytorch

[![docs](https://github.com/torchcvnn/torchcvnn/actions/workflows/doc.yml/badge.svg)](https://torchcvnn.github.io/torchcvnn/) ![pytest](https://github.com/torchcvnn/torchcvnn/actions/workflows/test.yml/badge.svg) [![PyPI version](https://badge.fury.io/py/torchcvnn.svg)](https://badge.fury.io/py/torchcvnn)

This is a library that uses [pytorch](https://pytorch.org) as a back-end for complex valued neural networks.

It was initially developed by Victor Dhédin and Jérémie Levi during their third year project at CentraleSupélec. 

## Installation

To install the library, it is simple as :

```
python -m pip install torchcvnn
```

## Installation for developping

To install when developping the library, within a virtual envrionment, you can :

```
git clone git@github.com:torchcvnn/torchcvnn.git
python3 -m venv torchcvnn-venv
source torchcvnn-venv/bin/activate
python -m pip install -e torchcvnn
```

This will install torchcvnn in developper mode. 

## Releasing a new version

To trigger the pipeline for a new release, you have to tag a commit and to push
it on the main branch.

**Update the version** First, do not forget to update the version in [src/torchcvnn/__init__.py]

**Trigger the CI pipeline for release** :

```
[main] git tag x.x.x
[main] git push --tags
```

This will trigger the `ci-cd.yml` pipeline which builds the distribution,
release it on github and on pypi.

Any commit that is not explicitely tagged with a version number does not trigger
the release ci-cd pipeline.

## Other projects

You might also be interested in some other projects: 

Tensorflow based : 

- [cvnn](https://github.com/NEGU93/cvnn) developed by colleagues from CentraleSupélec

Pytorch based : 

- [cplxmodule](https://github.com/ivannz/cplxmodule)
- [complexPyTorch](https://github.com/wavefrontshaping/complexPyTorch)
