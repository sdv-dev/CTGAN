# -*- coding: utf-8 -*-

"""Top-level package for ctgan."""

__author__ = 'MIT Data To AI Lab'
__email__ = 'dailabmit@gmail.com'
__version__ = '0.4.3'

from ctgan.demo import load_demo
from ctgan.synthesizers.ctgan import CTGANSynthesizer
from ctgan.synthesizers.tvae import TVAESynthesizer

__all__ = (
    'CTGANSynthesizer',
    'TVAESynthesizer',
    'load_demo'
)
