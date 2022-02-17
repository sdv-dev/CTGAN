#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Integration tests for ctgan.

These tests only ensure that the software does not crash and that
the API works as expected in terms of input and output data formats,
but correctness of the data values and the internal behavior of the
model are not checked.
"""

import tempfile as tf

import numpy as np
import pandas as pd
import pytest

from ctgan.synthesizers.ctgan import CTGANSynthesizer


def test_ctgan_no_categoricals():
    """Test the CTGANSynthesizer with no categorical values."""
    data = pd.DataFrame({
        'continuous': np.random.random(1000)
    })

    ctgan = CTGANSynthesizer(epochs=1)
    ctgan.fit(data, [])

    sampled = ctgan.sample(100)

    assert sampled.shape == (100, 1)
    assert isinstance(sampled, pd.DataFrame)
    assert set(sampled.columns) == {'continuous'}


def test_ctgan_dataframe():
    """Test the CTGANSynthesizer when passed a dataframe."""
    data = pd.DataFrame({
        'continuous': np.random.random(100),
        'discrete': np.random.choice(['a', 'b', 'c'], 100)
    })
    discrete_columns = ['discrete']

    ctgan = CTGANSynthesizer(epochs=1)
    ctgan.fit(data, discrete_columns)

    sampled = ctgan.sample(100)

    assert sampled.shape == (100, 2)
    assert isinstance(sampled, pd.DataFrame)
    assert set(sampled.columns) == {'continuous', 'discrete'}
    assert set(sampled['discrete'].unique()) == {'a', 'b', 'c'}


def test_ctgan_numpy():
    """Test the CTGANSynthesizer when passed a numpy array."""
    data = pd.DataFrame({
        'continuous': np.random.random(100),
        'discrete': np.random.choice(['a', 'b', 'c'], 100)
    })
    discrete_columns = [1]

    ctgan = CTGANSynthesizer(epochs=1)
    ctgan.fit(data.to_numpy(), discrete_columns)

    sampled = ctgan.sample(100)

    assert sampled.shape == (100, 2)
    assert isinstance(sampled, np.ndarray)
    assert set(np.unique(sampled[:, 1])) == {'a', 'b', 'c'}


def test_log_frequency():
    """Test the CTGANSynthesizer with no `log_frequency` set to False."""
    data = pd.DataFrame({
        'continuous': np.random.random(1000),
        'discrete': np.repeat(['a', 'b', 'c'], [950, 25, 25])
    })

    discrete_columns = ['discrete']

    ctgan = CTGANSynthesizer(epochs=100)
    ctgan.fit(data, discrete_columns)

    sampled = ctgan.sample(10000)
    counts = sampled['discrete'].value_counts()
    assert counts['a'] < 6500

    ctgan = CTGANSynthesizer(log_frequency=False, epochs=100)
    ctgan.fit(data, discrete_columns)

    sampled = ctgan.sample(10000)
    counts = sampled['discrete'].value_counts()
    assert counts['a'] > 9000


def test_categorical_nan():
    """Test the CTGANSynthesizer with no categorical values."""
    data = pd.DataFrame({
        'continuous': np.random.random(30),
        # This must be a list (not a np.array) or NaN will be cast to a string.
        'discrete': [np.nan, 'b', 'c'] * 10
    })
    discrete_columns = ['discrete']

    ctgan = CTGANSynthesizer(epochs=1)
    ctgan.fit(data, discrete_columns)

    sampled = ctgan.sample(100)

    assert sampled.shape == (100, 2)
    assert isinstance(sampled, pd.DataFrame)
    assert set(sampled.columns) == {'continuous', 'discrete'}

    # since np.nan != np.nan, we need to be careful here
    values = set(sampled['discrete'].unique())
    assert len(values) == 3
    assert any(pd.isna(x) for x in values)
    assert {'b', 'c'}.issubset(values)


def test_synthesizer_sample():
    """Test the CTGANSynthesizer samples the correct datatype."""
    data = pd.DataFrame({
        'discrete': np.random.choice(['a', 'b', 'c'], 100)
    })
    discrete_columns = ['discrete']

    ctgan = CTGANSynthesizer(epochs=1)
    ctgan.fit(data, discrete_columns)

    samples = ctgan.sample(1000, 'discrete', 'a')
    assert isinstance(samples, pd.DataFrame)


def test_save_load():
    """Test the CTGANSynthesizer load/save methods."""
    data = pd.DataFrame({
        'continuous': np.random.random(100),
        'discrete': np.random.choice(['a', 'b', 'c'], 100)
    })
    discrete_columns = ['discrete']

    ctgan = CTGANSynthesizer(epochs=1)
    ctgan.fit(data, discrete_columns)

    with tf.TemporaryDirectory() as temporary_directory:
        ctgan.save(temporary_directory + 'test_tvae.pkl')
        ctgan = CTGANSynthesizer.load(temporary_directory + 'test_tvae.pkl')

    sampled = ctgan.sample(1000)
    assert set(sampled.columns) == {'continuous', 'discrete'}
    assert set(sampled['discrete'].unique()) == {'a', 'b', 'c'}


def test_wrong_discrete_columns_dataframe():
    """Test the CTGANSynthesizer correctly crashes when passed non-existing discrete columns."""
    data = pd.DataFrame({
        'discrete': ['a', 'b']
    })
    discrete_columns = ['b', 'c']

    ctgan = CTGANSynthesizer(epochs=1)
    with pytest.raises(ValueError, match="Invalid columns found: {'.*', '.*'}"):
        ctgan.fit(data, discrete_columns)


def test_wrong_discrete_columns_numpy():
    """Test the CTGANSynthesizer correctly crashes when passed non-existing discrete columns."""
    data = pd.DataFrame({
        'discrete': ['a', 'b']
    })
    discrete_columns = [0, 1]

    ctgan = CTGANSynthesizer(epochs=1)
    with pytest.raises(ValueError, match=r'Invalid columns found: \[1\]'):
        ctgan.fit(data.to_numpy(), discrete_columns)


def test_wrong_sampling_conditions():
    """Test the CTGANSynthesizer correctly crashes when passed incorrect sampling conditions."""
    data = pd.DataFrame({
        'continuous': np.random.random(100),
        'discrete': np.random.choice(['a', 'b', 'c'], 100)
    })
    discrete_columns = ['discrete']

    ctgan = CTGANSynthesizer(epochs=1)
    ctgan.fit(data, discrete_columns)

    with pytest.raises(ValueError, match="The column_name `cardinal` doesn't exist in the data."):
        ctgan.sample(1, 'cardinal', "doesn't matter")

    with pytest.raises(ValueError):  # noqa: RDT currently incorrectly raises a tuple instead of a string
        ctgan.sample(1, 'discrete', 'd')


def test_fixed_random_seed():
    """Test the CTGANSynthesizer with a fixed seed.

    Expect that when the random seed is reset with the same seed, the same sequence
    of data will be produced. Expect that the data generated with the seed is
    different than randomly sampled data.
    """
    # Setup
    data = pd.DataFrame({
        'continuous': np.random.random(100),
        'discrete': np.random.choice(['a', 'b', 'c'], 100)
    })
    discrete_columns = ['discrete']

    ctgan = CTGANSynthesizer(epochs=1)

    # Run
    ctgan.fit(data, discrete_columns)
    sampled_random = ctgan.sample(10)

    ctgan.set_random_state(0)
    sampled_0_0 = ctgan.sample(10)
    sampled_0_1 = ctgan.sample(10)

    ctgan.set_random_state(0)
    sampled_1_0 = ctgan.sample(10)
    sampled_1_1 = ctgan.sample(10)

    # Assert
    assert not np.array_equal(sampled_random, sampled_0_0)
    assert not np.array_equal(sampled_random, sampled_0_1)
    np.testing.assert_array_equal(sampled_0_0, sampled_1_0)
    np.testing.assert_array_equal(sampled_0_1, sampled_1_1)


# Below are CTGAN tests that should be implemented in the future
def test_continuous():
    """Test training the CTGAN synthesizer on a continuous dataset."""
    # assert the distribution of the samples is close to the distribution of the data
    # using kstest:
    #   - uniform (assert p-value > 0.05)
    #   - gaussian (assert p-value > 0.05)
    #   - inversely correlated (assert correlation < 0)


def test_categorical():
    """Test training the CTGAN synthesizer on a categorical dataset."""
    # assert the distribution of the samples is close to the distribution of the data
    # using cstest:
    #   - uniform (assert p-value > 0.05)
    #   - very skewed / biased? (assert p-value > 0.05)
    #   - inversely correlated (assert correlation < 0)


def test_categorical_log_frequency():
    """Test training the CTGAN synthesizer on a small categorical dataset."""
    # assert the distribution of the samples is close to the distribution of the data
    # using cstest:
    #   - uniform (assert p-value > 0.05)
    #   - very skewed / biased? (assert p-value > 0.05)
    #   - inversely correlated (assert correlation < 0)


def test_mixed():
    """Test training the CTGAN synthesizer on a small mixed-type dataset."""
    # assert the distribution of the samples is close to the distribution of the data
    # using a kstest for continuous + a cstest for categorical.


def test_conditional():
    """Test training the CTGAN synthesizer and sampling conditioned on a categorical."""
    # verify that conditioning increases the likelihood of getting a sample with the specified
    # categorical value


def test_batch_size_pack_size():
    """Test that if batch size is not a multiple of pack size, it raises a sane error."""
