<div align="center">
<br/>
<p align="center">
    <i>This repository is part of <a href="https://sdv.dev">The Synthetic Data Vault Project</a>, a project from <a href="https://datacebo.com">DataCebo</a>.</i>
</p>

[![Development Status](https://img.shields.io/badge/Development%20Status-2%20--%20Pre--Alpha-yellow)](https://pypi.org/search/?c=Development+Status+%3A%3A+2+-+Pre-Alpha)
[![PyPI Shield](https://img.shields.io/pypi/v/ctgan.svg)](https://pypi.python.org/pypi/ctgan)
[![Unit Tests](https://github.com/sdv-dev/CTGAN/actions/workflows/unit.yml/badge.svg)](https://github.com/sdv-dev/CTGAN/actions/workflows/unit.yml)
[![Downloads](https://pepy.tech/badge/ctgan)](https://pepy.tech/project/ctgan)
[![Coverage Status](https://codecov.io/gh/sdv-dev/CTGAN/branch/master/graph/badge.svg)](https://codecov.io/gh/sdv-dev/CTGAN)

<div align="left">
<br/>
<p align="center">
<a href="https://github.com/sdv-dev/CTGAN">
<img align="center" width=40% src="https://github.com/sdv-dev/SDV/blob/master/docs/images/CTGAN-DataCebo.png"></img>
</a>
</p>
</div>

</div>

# Overview

CTGAN is a collection of Deep Learning based Synthetic Data Generators for single table data, which are able to learn from real data and generate synthetic clones with high fidelity.

| Important Links                               |                                                                      |
| --------------------------------------------- | -------------------------------------------------------------------- |
| :computer: **[Website]**                      | Check out the SDV Website for more information about the project.    |
| :orange_book: **[SDV Blog]**                  | Regular publshing of useful content about Synthetic Data Generation. |
| :book: **[Documentation]**                    | Quickstarts, User and Development Guides, and API Reference.         |
| :octocat: **[Repository]**                    | The link to the Github Repository of this library.                   |
| :scroll: **[License]**                        | The entire ecosystem is published under the MIT License.             |
| :keyboard: **[Development Status]**           | This software is in its Pre-Alpha stage.                             |
| [![][Slack Logo] **Community**][Community]    | Join our Slack Workspace for announcements and discussions.          |
| [![][MyBinder Logo] **Tutorials**][Tutorials] | Run the SDV Tutorials in a Binder environment.                       |

[Website]: https://sdv.dev
[SDV Blog]: https://sdv.dev/blog
[Documentation]: https://sdv.dev/SDV
[Repository]: https://github.com/sdv-dev/CTGAN
[License]: https://github.com/sdv-dev/CTGAN/blob/master/LICENSE
[Development Status]: https://pypi.org/search/?c=Development+Status+%3A%3A+2+-+Pre-Alpha
[Slack Logo]: https://github.com/sdv-dev/SDV/blob/master/docs/images/slack.png
[Community]: https://join.slack.com/t/sdv-space/shared_invite/zt-gdsfcb5w-0QQpFMVoyB2Yd6SRiMplcw
[MyBinder Logo]: https://github.com/sdv-dev/SDV/blob/master/docs/images/mybinder.png
[Tutorials]: https://mybinder.org/v2/gh/sdv-dev/SDV/master?filepath=tutorials

## Implemented Models

Currently, this library implements the **CTGAN** and **TVAE** models proposed in the [Modeling Tabular data using Conditional GAN](https://arxiv.org/abs/1907.00503) paper. For more information about these models, please check out the respective user guides:
* [CTGAN User Guide](https://sdv.dev/SDV/user_guides/single_table/ctgan.html).
* [TVAE User Guide](https://sdv.dev/SDV/user_guides/single_table/tvae.html).

# Install

**CTGAN** is part of the **SDV** project and is automatically installed alongside it. For
details about this process please visit the [SDV Installation Guide](
https://sdv.dev/SDV/getting_started/install.html)

Optionally, **CTGAN** can also be installed as a standalone library using the following commands:

**Using `pip`:**

```bash
pip install ctgan
```

**Using `conda`:**

```bash
conda install -c pytorch -c conda-forge ctgan
```

For more installation options please visit the [CTGAN installation Guide](INSTALL.md)

# Usage Example

> :warning: **WARNING**: If you're just getting started with synthetic data, we recommend using the SDV library which provides user-friendly APIs for interacting with CTGAN. To learn more about using CTGAN through SDV, check out the user guide [here](https://sdv.dev/SDV/user_guides/single_table/ctgan.html).

To get started with CTGAN, you should prepare your data as either a `numpy.ndarray` or a `pandas.DataFrame` object with two types of columns:

* **Continuous Columns**: can contain any numerical value.
* **Discrete Columns**: contain a finite number values, whether these are string values or not.

In this example we load the [Adult Census Dataset](https://archive.ics.uci.edu/ml/datasets/adult) which is a built-in demo dataset. We then model it using the **CTGANSynthesizer** and generate a synthetic copy of it.


```python3
from ctgan import CTGANSynthesizer
from ctgan import load_demo

data = load_demo()

# Names of the columns that are discrete
discrete_columns = [
    'workclass',
    'education',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'native-country',
    'income'
]

ctgan = CTGANSynthesizer(epochs=10)
ctgan.fit(data, discrete_columns)

# Synthetic copy
samples = ctgan.sample(1000)
```



# Join our community


1. Please have a look at the [Contributing Guide](https://sdv.dev/SDV/developer_guides/contributing.html) to see how you can contribute to the project.
2. If you have any doubts, feature requests or detect an error, please [open an issue on github](https://github.com/sdv-dev/CTGAN/issues) or [join our Slack Workspace](https://sdv-space.slack.com/join/shared_invite/zt-gdsfcb5w-0QQpFMVoyB2Yd6SRiMplcw#/).
3. Also, do not forget to check the [project documentation site](https://sdv.dev/SDV/)!


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

# Related Projects
Please note that these libraries are external contributions and are not maintained nor supervised by
the MIT DAI-Lab team.

## R interface for CTGAN

A wrapper around **CTGAN** has been implemented by Kevin Kuo @kevinykuo, bringing the functionalities
of **CTGAN** to **R** users.

More details can be found in the corresponding repository: https://github.com/kasaai/ctgan

## CTGAN Server CLI

A package to easily deploy **CTGAN** onto a remote server. This package is developed by Timothy Pillow @oregonpillow.

More details can be found in the corresponding repository: https://github.com/oregonpillow/ctgan-server-cli

---


<div align="center">
<a href="https://datacebo.com"><img align="center" width=40% src="https://github.com/sdv-dev/SDV/blob/master/docs/images/DataCebo.png"></img></a>
</div>
<br/>
<br/>

[The Synthetic Data Vault Project](https://sdv.dev) was first created at MIT's [Data to AI Lab](
https://dai.lids.mit.edu/) in 2016. After 4 years of research and traction with enterprise, we
created [DataCebo](https://datacebo.com) in 2020 with the goal of growing the project.
Today, DataCebo is the proud developer of SDV, the largest ecosystem for
synthetic data generation & evaluation. It is home to multiple libraries that support synthetic
data, including:

* 🔄 Data discovery & transformation. Reverse the transforms to reproduce realistic data.
* 🧠 Multiple machine learning models -- ranging from Copulas to Deep Learning -- to create tabular,
  multi table and time series data.
* 📊 Measuring quality and privacy of synthetic data, and comparing different synthetic data
  generation models.

[Get started using the SDV package](https://sdv.dev/SDV/getting_started/install.html) -- a fully
integrated solution and your one-stop shop for synthetic data. Or, use the standalone libraries
for specific needs.
