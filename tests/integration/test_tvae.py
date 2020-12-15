#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Integration tests for tvae.

These tests only ensure that the software does not crash and that
the API works as expected in terms of input and output data formats,
but correctness of the data values and the internal behavior of the
model are not checked.
"""

import numpy as np
import pandas as pd

from ctgan.synthesizers.tvae import TVAESynthesizer


def test_tvae_dataframe():
    data = pd.DataFrame({
        'continuous': np.random.random(1000),
        'discrete': np.random.choice(['a', 'b'], 1000)
    })
    discrete_columns = ['discrete']

    tvae = TVAESynthesizer(epochs=10)
    tvae.fit(data, discrete_columns)

    sampled = tvae.sample(100)

    assert sampled.shape == (100, 2)
    assert isinstance(sampled, pd.DataFrame)
    assert set(sampled.columns) == {'continuous', 'discrete'}
    assert set(sampled['discrete'].unique()) == {'a', 'b'}


def test_tvae_numpy():
    data = pd.DataFrame({
        'continuous': np.random.random(1000),
        'discrete': np.random.choice(['a', 'b'], 1000)
    })
    discrete_columns = [1]

    tvae = TVAESynthesizer(epochs=10)
    tvae.fit(data.values, discrete_columns)

    sampled = tvae.sample(100)

    assert sampled.shape == (100, 2)
    assert isinstance(sampled, np.ndarray)
    assert set(np.unique(sampled[:, 1])) == {'a', 'b'}


def test_synthesizer_sample():
    data = pd.DataFrame({
        'discrete': np.random.choice(['a', 'b'], 100)
    })
    discrete_columns = ['discrete']

    tvae = TVAESynthesizer(epochs=1)
    tvae.fit(data, discrete_columns)

    samples = tvae.sample(1000)
    assert isinstance(samples, pd.DataFrame)


def test_save_load():
    data = pd.DataFrame({
        'continuous': np.random.random(100),
        'discrete': np.random.choice(['a', 'b'], 100)
    })
    discrete_columns = ['discrete']

    tvae = TVAESynthesizer(epochs=1)
    tvae.fit(data, discrete_columns)
    tvae.save("test_tvae.pkl")

    tvae = TVAESynthesizer.load("test_tvae.pkl")

    sampled = tvae.sample(1000)
    assert set(sampled.columns) == {'continuous', 'discrete'}
    assert set(sampled['discrete'].unique()) == {'a', 'b'}
