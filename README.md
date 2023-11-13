<div align="center">
<br/>
<p align="center">
    <i>This repository is part of <a href="https://sdv.dev">The Synthetic Data Vault Project</a>, a project from <a href="https://datacebo.com">DataCebo</a>.</i>
</p>

[![Development Status](https://img.shields.io/badge/Development%20Status-2%20--%20Pre--Alpha-yellow)](https://pypi.org/search/?c=Development+Status+%3A%3A+2+-+Pre-Alpha)
[![PyPI Shield](https://img.shields.io/pypi/v/ctgan.svg)](https://pypi.python.org/pypi/ctgan)
[![Unit Tests](https://github.com/sdv-dev/CTGAN/actions/workflows/unit.yml/badge.svg)](https://github.com/sdv-dev/CTGAN/actions/workflows/unit.yml)
[![Downloads](https://pepy.tech/badge/ctgan)](https://pepy.tech/project/ctgan)
[![Coverage Status](https://codecov.io/gh/sdv-dev/CTGAN/branch/main/graph/badge.svg)](https://codecov.io/gh/sdv-dev/CTGAN)

<div align="left">
<br/>
<p align="center">
<a href="https://github.com/sdv-dev/CTGAN">
<img align="center" width=40% src="https://github.com/sdv-dev/SDV/blob/stable/docs/images/CTGAN-DataCebo.png"></img>
</a>
</p>
</div>

</div>

# Overview

CTGAN is a collection of Deep Learning based synthetic data generators for single table data, which are able to learn from real data and generate synthetic data with high fidelity.

| Important Links                               |                                                                      |
| --------------------------------------------- | -------------------------------------------------------------------- |
| :computer: **[Website]**                      | Check out the SDV Website for more information about our overall synthetic data ecosystem.|
| :orange_book: **[Blog]**                      | A deeper look at open source, synthetic data creation and evaluation.|
| :book: **[Documentation]**                    | Quickstarts, User and Development Guides, and API Reference.         |
| :octocat: **[Repository]**                    | The link to the Github Repository of this library.                   |
| :keyboard: **[Development Status]**           | This software is in its Pre-Alpha stage.                             |
| [![][Slack Logo] **Community**][Community]    | Join our Slack Workspace for announcements and discussions.          |

[Website]: https://sdv.dev
[Blog]: https://datacebo.com/blog
[Documentation]: https://bit.ly/sdv-docs
[Repository]: https://github.com/sdv-dev/CTGAN
[License]: https://github.com/sdv-dev/CTGAN/blob/main/LICENSE
[Development Status]: https://pypi.org/search/?c=Development+Status+%3A%3A+2+-+Pre-Alpha
[Slack Logo]: https://github.com/sdv-dev/SDV/blob/stable/docs/images/slack.png
[Community]: https://bit.ly/sdv-slack-invite

Currently, this library implements the **CTGAN** and **TVAE** models described in the [Modeling Tabular data using Conditional GAN](https://arxiv.org/abs/1907.00503) paper, presented at the 2019 NeurIPS conference.

# Install

## Use CTGAN through the SDV library

:warning: If you're just getting started with synthetic data, we recommend installing the SDV library which provides user-friendly APIs for accessing CTGAN. :warning:

The SDV library provides wrappers for preprocessing your data as well as additional usability features like constraints. See the [SDV documentation](https://bit.ly/sdv-docs) to get started.

## Use the CTGAN standalone library

Alternatively, you can also install and use **CTGAN** directly, as a standalone library:

**Using `pip`:**

```bash
pip install ctgan
```

**Using `conda`:**

```bash
conda install -c pytorch -c conda-forge ctgan
```

When using the CTGAN library directly, you may need to manually preprocess your data into the correct format, for example:

* Continuous data must be represented as floats
* Discrete data must be represented as ints or strings
* The data should not contain any missing values

# Usage Example

In this example we load the [Adult Census Dataset](https://archive.ics.uci.edu/ml/datasets/adult)* which is a built-in demo dataset. We use CTGAN to learn from the real data and then generate some synthetic data.

```python3
from ctgan import CTGAN
from ctgan import load_demo

real_data = load_demo()

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

ctgan = CTGAN(epochs=10)
ctgan.fit(real_data, discrete_columns)

# Create synthetic data
synthetic_data = ctgan.sample(1000)
```

*For more information about the dataset see:
Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml].
Irvine, CA: University of California, School of Information and Computer Science.

# Join our community

Join our [Slack channel](https://bit.ly/sdv-slack-invite) to discuss more about CTGAN and synthetic data. If you find a bug or have a feature request, you can also [open an issue](https://github.com/sdv-dev/CTGAN/issues) on our GitHub.

**Interested in contributing to CTGAN?** Read our [Contribution Guide](CONTRIBUTING.rst) to get started.

# Citing CTGAN

If you use CTGAN, please cite the following work:

*Lei Xu, Maria Skoularidou, Alfredo Cuesta-Infante, Kalyan Veeramachaneni.* **Modeling Tabular data using Conditional GAN**. NeurIPS, 2019.

```LaTeX
@inproceedings{ctgan,
  title={Modeling Tabular data using Conditional GAN},
  author={Xu, Lei and Skoularidou, Maria and Cuesta-Infante, Alfredo and Veeramachaneni, Kalyan},
  booktitle={Advances in Neural Information Processing Systems},
  year={2019}
}
```

# Related Projects
Please note that these projects are external to the SDV Ecosystem. They are not affiliated with or maintained by DataCebo.

* **R Interface for CTGAN**: A wrapper around **CTGAN** that brings the functionalities to **R** users.
More details can be found in the corresponding repository: https://github.com/kasaai/ctgan
* **CTGAN Server CLI**: A package to easily deploy CTGAN onto a remote server. Created by Timothy Pillow @oregonpillow at: https://github.com/oregonpillow/ctgan-server-cli

---


<div align="center">
<a href="https://datacebo.com"><img align="center" width=40% src="https://github.com/sdv-dev/SDV/blob/stable/docs/images/DataCebo.png"></img></a>
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
