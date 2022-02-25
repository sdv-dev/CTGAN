#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import find_packages, setup

with open('README.md', encoding='utf-8') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md', encoding='utf-8') as history_file:
    history = history_file.read()

install_requires = [
    'packaging>=20,<22',
    "numpy>=1.18.0,<1.20.0;python_version<'3.7'",
    "numpy>=1.20.0,<2;python_version>='3.7'",
    'pandas>=1.1.3,<2',
    'scikit-learn>=0.24,<2',
    'torch>=1.8.0,<2',
    'torchvision>=0.9.0,<1',
    'rdt>=0.6.2,<0.7',
]

setup_requires = [
    'pytest-runner>=2.11.1',
]

tests_require = [
    'pytest>=3.4.2',
    'pytest-rerunfailures>=9.1.1,<10',
    'pytest-cov>=2.6.0',
]

development_requires = [
    # general
    'pip>=9.0.1',
    'bumpversion>=0.5.3,<0.6',
    'watchdog>=0.8.3,<0.11',

    # style check
    'flake8>=3.7.7,<4',
    'isort>=4.3.4,<5',
    'dlint>=0.11.0,<0.12',  # code security addon for flake8
    'flake8-debugger>=4.0.0,<4.1',
    'flake8-mock>=0.3,<0.4',
    'flake8-mutable>=1.2.0,<1.3',
    'flake8-absolute-import>=1.0,<2',
    'flake8-multiline-containers>=0.0.18,<0.1',
    'flake8-print>=4.0.0,<4.1',
    'flake8-quotes>=3.3.0,<4',
    'flake8-fixme>=1.1.1,<1.2',
    'flake8-expression-complexity>=0.0.9,<0.1',
    'flake8-eradicate>=1.1.0,<1.2',
    'flake8-builtins>=1.5.3,<1.6',
    'flake8-variables-names>=0.0.4,<0.1',
    'pandas-vet>=0.2.2,<0.3',
    'flake8-comprehensions>=3.6.1,<3.7',
    'dlint>=0.11.0,<0.12',
    'flake8-docstrings>=1.5.0,<2',
    'flake8-sfs>=0.0.3,<0.1',
    'flake8-pytest-style>=1.5.0,<2',

    # fix style issues
    'autoflake>=1.1,<2',
    'autopep8>=1.4.3,<1.6',

    # distribute on PyPI
    'twine>=1.10.0,<4',
    'wheel>=0.30.0',

    # Advanced testing
    'coverage>=4.5.1,<6',
    'tox>=2.9.1,<4',

    'invoke',
]

setup(
    author='MIT Data To AI Lab',
    author_email='dailabmit@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    description='Conditional GAN for Tabular Data',
    entry_points={
        'console_scripts': [
            'ctgan=ctgan.__main__:main'
        ],
    },
    extras_require={
        'test': tests_require,
        'dev': development_requires + tests_require,
    },
    install_package_data=True,
    install_requires=install_requires,
    license='MIT license',
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/markdown',
    include_package_data=True,
    keywords='ctgan CTGAN',
    name='ctgan',
    packages=find_packages(include=['ctgan', 'ctgan.*']),
    python_requires='>=3.6,<3.10',
    setup_requires=setup_requires,
    test_suite='tests',
    tests_require=tests_require,
    url='https://github.com/sdv-dev/CTGAN',
    version='0.5.2.dev0',
    zip_safe=False,
)
