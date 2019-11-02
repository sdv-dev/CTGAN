[![PyPI Shield](https://img.shields.io/pypi/v/ctgan.svg)](https://pypi.python.org/pypi/ctgan)
[![Travis CI Shield](https://travis-ci.org/DAI-Lab/ctgan.svg?branch=master)](https://travis-ci.org/DAI-Lab/ctgan)

# ctgan

Implementation of our NeurIPS paper **Modeling Tabular data using Conditional GAN**. 

CTGAN is a GAN-based data synthesizer that can generate synthetic tabular data with high fidelity. 

- Free software: MIT license
- Documentation: https://DAI-Lab.github.io/ctgan
- Homepage: https://github.com/DAI-Lab/ctgan


## Overview

Based on previous work ([TGAN](https://github.com/DAI-Lab/tgan)) on synthetic data generation, we develop a new model called CTGAN. Several major differences make CTGAN outperform TGAN.

- **Preprocessing**: CTGAN uses more sophisticated Variational Gaussian Mixture Model to detect modes of continuous columns. 
- **Network structure**: TGAN uses LSTM to generate synthetic data column by column. CTGAN uses Fully-connected networks which is more efficient. 
- **Features to prevent mode collapse**: We design a conditional generator and resample the training data to prevent model collapse on discrete columns. We use WGANGP and PacGAN to stabilize the training of GAN.

## Requirements
- Python 3.5, 3.6, 3.7
- Pytorch >= 1.0
- sklearn
- numpy
- pandas
- absl-py

## Install
### Pip install
```
> pip install git+https://github.com/DAI-Lab/ctgan@master
> python3 -m ctgan.cli [...]
```

### Run without install
Make sure all the required libraries are installed. 

```
> git clone https://github.com/DAI-Lab/ctgan
> cd ctgan
> python3 -m ctgan.cli [...]
```

## Tutorial
### Data format
The data is a space (or tab) separated file. For example,

```
100        A        True
200        B        False
105        A        True
120        C        False
...        ...        ...
```


Metafile describes each row as one line. `C` or `D` at the beginning of each line represent continuous column or discrete column respectively. For continuous column, the following two number indicates the range of the column. For discrete column, the following strings indicate all possible values in the column. For example,

```
C    0    500
D    A    B    C
D    True     False
```


### Run model
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

### Example
It's easy to try our model using example datasets. 

```
git clone https://github.com/DAI-Lab/ctgan
cd ctgan
python3 -m ctgan.cli --data example/adult.dat --meta example/adult.meta

```

## Citing TGAN

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
