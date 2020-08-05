# -*- coding: utf-8 -*-

"""Top-level package for ctgan."""

__author__ = 'MIT Data To AI Lab'
__email__ = 'dailabmit@gmail.com'
__version__ = '0.3.0.dev0'

from ctgan.demo import load_demo
from ctgan.synthesizer import CTGANSynthesizer

__all__ = (
    'CTGANSynthesizer',
    'load_demo'
)
