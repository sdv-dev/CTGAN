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


def test_log_frequency():

    data = pd.DataFrame({
        'continuous': np.random.random(1000),
        'discrete': np.repeat(['a', 'b', 'c'], [950, 25, 25])
    })

    discrete_columns = ['discrete']

    ctgan = CTGANSynthesizer()
    ctgan.fit(data, discrete_columns, epochs=100)

    sampled = ctgan.sample(10000)
    counts = sampled['discrete'].value_counts()
    assert counts['a'] < 6500

    ctgan = CTGANSynthesizer()
    ctgan.fit(data, discrete_columns, epochs=100, log_frequency=False)

    sampled = ctgan.sample(10000)
    counts = sampled['discrete'].value_counts()
    assert counts['a'] > 9000


def test_categorical_nan():
    data = pd.DataFrame({
        'continuous': np.random.random(30),
        # This must be a list (not a np.array) or NaN will be cast to a string.
        'discrete': [np.nan, 'b', 'c'] * 10
    })
    discrete_columns = ['discrete']

    ctgan = CTGANSynthesizer()
    ctgan.fit(data, discrete_columns, epochs=1)

    sampled = ctgan.sample(100)

    assert sampled.shape == (100, 2)
    assert isinstance(sampled, pd.DataFrame)
    assert set(sampled.columns) == {'continuous', 'discrete'}

    # since np.nan != np.nan, we need to be careful here
    values = set(sampled['discrete'].unique())
    assert len(values) == 3
    assert any(pd.isnull(x) for x in values)
    assert {"b", "c"}.issubset(values)
