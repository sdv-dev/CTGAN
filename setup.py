#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md') as history_file:
    history = history_file.read()

requirements = ['Click>=6.0',
                'torch',
                'torchvision',
                'sklearn',
                'numpy',
                'pandas',
                'absl-py'
]

setup_requirements = [
    'pytest-runner>=2.11.1',
]

test_requirements = [
    'coverage>=4.5.1',
    'pytest>=3.4.2',
    'tox>=2.9.1',
]

development_requirements = [
    # general
    'bumpversion>=0.5.3',
    'pip>=9.0.1',
    'watchdog>=0.8.3',

    # docs
    'm2r>=0.2.0',
    'Sphinx>=1.7.1',
    'sphinx_rtd_theme>=0.2.4',

    # style check
    'flake8>=3.5.0',
    'isort>=4.3.4',

    # fix style issues
    'autoflake>=1.1',
    'autopep8>=1.3.5',

    # distribute on PyPI
    'twine>=1.10.0',
    'wheel>=0.30.0',
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
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description='CTGAN is a GAN-based data synthesizer that can generate synthetic tabular data with high fidelity. ',
    entry_points={
        'console_scripts': [
            'ctgan=ctgan.cli:main',
        ],
    },
    extras_require={
        'test': test_requirements,
        'dev': development_requirements + test_requirements,
    },
    install_package_data=True,
    install_requires=requirements,
    license='MIT license',
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/markdown',
    include_package_data=True,
    keywords='ctgan',
    name='ctgan',
    packages=find_packages(include=['ctgan', 'ctgan.*']),
    python_requires='>=3.4',
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/DAI-Lab/ctgan',
    version='0.1.0',
    zip_safe=False,
)
