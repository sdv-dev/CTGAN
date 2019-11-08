<p align="left">
<img width=15% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt=“DAI-Lab” />
<i>An open source project from Data to AI Lab at MIT.</i>
</p>

[![PyPI Shield](https://img.shields.io/pypi/v/ctgan.svg)](https://pypi.python.org/pypi/ctgan)
[![Travis CI Shield](https://travis-ci.org/DAI-Lab/CTGAN.svg?branch=master)](https://travis-ci.org/DAI-Lab/CTGAN)

<!--[![Downloads](https://pepy.tech/badge/ctgan)](https://pepy.tech/project/ctgan)-->

# CTGAN

Implementation of our NeurIPS paper **Modeling Tabular data using Conditional GAN**.

CTGAN is a GAN-based data synthesizer that can generate synthetic tabular data with high fidelity.

- Free software: MIT license
- Documentation: https://DAI-Lab.github.io/CTGAN
- Homepage: https://github.com/DAI-Lab/CTGAN

# Overview

Based on previous work ([TGAN](https://github.com/DAI-Lab/tgan)) on synthetic data generation, we develop a new model called CTGAN. Several major differences make CTGAN outperform TGAN.

- **Preprocessing**: CTGAN uses more sophisticated Variational Gaussian Mixture Model to detect modes of continuous columns.
- **Network structure**: TGAN uses LSTM to generate synthetic data column by column. CTGAN uses Fully-connected networks which is more efficient.
- **Features to prevent mode collapse**: We design a conditional generator and resample the training data to prevent model collapse on discrete columns. We use WGANGP and PacGAN to stabilize the training of GAN.

# Install

## Requirements

**CTGAN** has been developed and tested on [Python 3.5, 3.6 and 3.7](https://www.python.org/downloads/)

## Install from PyPI

The recommended way to installing **CTGAN** is using [pip](https://pip.pypa.io/en/stable/):

```bash
pip install ctgan
```

This will pull and install the latest stable release from [PyPI](https://pypi.org/).

## Install from source

Alternatively, you can clone the repository and install it from
source by running `make install` on the `stable` branch:

```bash
git clone git@github.com:DAI-Lab/CTGAN.git
cd CTGAN
git checkout stable
make install
```

## Install for Development

If you want to contribute to the project, a few more steps are required to make the project ready
for development.

Please head to the [Contributing Guide](https://DAI-Lab.github.io/CTGAN/contributing.html#get-started)
for more details about this process.

# Quickstart

In this short tutorial we will guide you through a series of steps that will help you
getting started with **CTGAN**.


## Data format

The data is a space (or tab) separated file. For example,

```
100        A        True
200        B        False
105        A        True
120        C        False
...        ...        ...
```


Metafile describes each column as one line. `C` or `D` at the beginning of each line represent continuous column or discrete column respectively. For continuous column, the following two number indicates the range of the column. For discrete column, the following strings indicate all possible values in the column. For example,

```
C    0    500
D    A    B    C
D    True     False
```

## Run model

```
USAGE:
    python3 ctgan/cli.py [flags]
flags:
  --data: Filename of training data.
    (default: '')
  --max_epoch: Epoches to train.
    (default: '100')
    (an integer)
  --meta: Filename of meta data.
    (default: '')
  --model_dir: Path to save model.
    (default: '')
  --output: Output filename.
    (default: '')
  --sample: Number of rows to generate.
    (default: '1000')
    (an integer)
```

## Example

It's easy to try our model using example datasets.

```
git clone https://github.com/DAI-Lab/ctgan
cd ctgan
python3 -m ctgan.cli --data examples/adult.dat --meta examples/adult.meta
```


## What's next?

For more details about **CTGAN** and all its possibilities
and features, please check the [documentation site](https://DAI-Lab.github.io/CTGAN/).


# Citing TGAN

If you use CTGAN, please cite the following work:

- *Lei Xu, Maria Skoularidou, Alfredo Cuesta-Infante, Kalyan Veeramachaneni.* **Modeling Tabular data using Conditional GAN**. NeurIPS, 2019.

```LaTeX
@inproceedings{xu2019modeling,
  title={Modeling Tabular data using Conditional GAN},
  author={Xu, Lei and Skoularidou, Maria and Cuesta-Infante, Alfredo and Veeramachaneni, Kalyan},
  booktitle={Advances in Neural Information Processing Systems},
  year={2019}
}
```
