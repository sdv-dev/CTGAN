#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Integration tests for ctgan.

These tests only ensure that the software does not crash and that
the API works as expected in terms of input and output data formats,
but correctness of the data values and the internal behavior of the
model are not checked.
"""

import numpy as np
import pandas as pd

from ctgan.synthesizer import CTGANSynthesizer


def test_ctgan_dataframe():
    data = pd.DataFrame({
        'continuous': np.random.random(100),
        'discrete': np.random.choice(['a', 'b', 'c'], 100)
    })
    discrete_columns = ['discrete']

    ctgan = CTGANSynthesizer()
    ctgan.fit(data, discrete_columns, epochs=1)

    sampled = ctgan.sample(100)

    assert sampled.shape == (100, 2)
    assert isinstance(sampled, pd.DataFrame)
    assert set(sampled.columns) == {'continuous', 'discrete'}
    assert set(sampled['discrete'].unique()) == {'a', 'b', 'c'}


def test_ctgan_numpy():
    data = pd.DataFrame({
        'continuous': np.random.random(100),
        'discrete': np.random.choice(['a', 'b', 'c'], 100)
    })
    discrete_columns = [1]

    ctgan = CTGANSynthesizer()
    ctgan.fit(data.values, discrete_columns, epochs=1)

    sampled = ctgan.sample(100)

    assert sampled.shape == (100, 2)
    assert isinstance(sampled, np.ndarray)
    assert set(np.unique(sampled[:, 1])) == {'a', 'b', 'c'}
