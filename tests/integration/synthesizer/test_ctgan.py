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

from ctgan.errors import InvalidDataError
from ctgan.synthesizers.ctgan import CTGAN


def test_ctgan_no_categoricals():
    """Test the CTGAN with no categorical values."""
    data = pd.DataFrame({'continuous': np.random.random(1000)})

    ctgan = CTGAN(epochs=1)
    ctgan.fit(data, [])

    sampled = ctgan.sample(100)

    assert sampled.shape == (100, 1)
    assert isinstance(sampled, pd.DataFrame)
    assert set(sampled.columns) == {'continuous'}
    assert len(ctgan.loss_values) == 1
    assert list(ctgan.loss_values.columns) == ['Epoch', 'Generator Loss', 'Discriminator Loss']


def test_ctgan_dataframe():
    """Test the CTGAN when passed a dataframe."""
    data = pd.DataFrame({
        'continuous': np.random.random(100),
        'discrete': np.random.choice(['a', 'b', 'c'], 100),
    })
    discrete_columns = ['discrete']

    ctgan = CTGAN(epochs=1)
    ctgan.fit(data, discrete_columns)

    sampled = ctgan.sample(100)

    assert sampled.shape == (100, 2)
    assert isinstance(sampled, pd.DataFrame)
    assert set(sampled.columns) == {'continuous', 'discrete'}
    assert set(sampled['discrete'].unique()) == {'a', 'b', 'c'}
    assert len(ctgan.loss_values) == 1
    assert list(ctgan.loss_values.columns) == ['Epoch', 'Generator Loss', 'Discriminator Loss']


def test_ctgan_numpy():
    """Test the CTGAN when passed a numpy array."""
    data = pd.DataFrame({
        'continuous': np.random.random(100),
        'discrete': np.random.choice(['a', 'b', 'c'], 100),
    })
    discrete_columns = [1]

    ctgan = CTGAN(epochs=1)
    ctgan.fit(data.to_numpy(), discrete_columns)

    sampled = ctgan.sample(100)

    assert sampled.shape == (100, 2)
    assert isinstance(sampled, np.ndarray)
    assert set(np.unique(sampled[:, 1])) == {'a', 'b', 'c'}
    assert len(ctgan.loss_values) == 1
    assert list(ctgan.loss_values.columns) == ['Epoch', 'Generator Loss', 'Discriminator Loss']


def test_log_frequency():
    """Test the CTGAN with no `log_frequency` set to False."""
    data = pd.DataFrame({
        'continuous': np.random.random(1000),
        'discrete': np.repeat(['a', 'b', 'c'], [950, 25, 25]),
    })

    discrete_columns = ['discrete']

    ctgan = CTGAN(epochs=100)
    ctgan.fit(data, discrete_columns)

    assert len(ctgan.loss_values) == 100
    assert list(ctgan.loss_values.columns) == ['Epoch', 'Generator Loss', 'Discriminator Loss']
    pd.testing.assert_series_equal(ctgan.loss_values['Epoch'], pd.Series(range(100), name='Epoch'))

    sampled = ctgan.sample(10000)
    counts = sampled['discrete'].value_counts()
    assert counts['a'] < 6500

    ctgan = CTGAN(log_frequency=False, epochs=100)
    ctgan.fit(data, discrete_columns)

    assert len(ctgan.loss_values) == 100
    assert list(ctgan.loss_values.columns) == ['Epoch', 'Generator Loss', 'Discriminator Loss']
    pd.testing.assert_series_equal(ctgan.loss_values['Epoch'], pd.Series(range(100), name='Epoch'))

    sampled = ctgan.sample(10000)
    counts = sampled['discrete'].value_counts()
    assert counts['a'] > 9000


def test_categorical_nan():
    """Test the CTGAN with no categorical values."""
    data = pd.DataFrame({
        'continuous': np.random.random(30),
        # This must be a list (not a np.array) or NaN will be cast to a string.
        'discrete': [np.nan, 'b', 'c'] * 10,
    })
    discrete_columns = ['discrete']

    ctgan = CTGAN(epochs=1)
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


def test_continuous_nan():
    """Test the CTGAN with missing numerical values."""
    # Setup
    data = pd.DataFrame({
        'continuous': [np.nan, 1.0, 2.0] * 10,
        'discrete': ['a', 'b', 'c'] * 10,
    })
    discrete_columns = ['discrete']
    error_message = (
        'CTGAN does not support null values in the continuous training data. '
        'Please remove all null values from your continuous training data.'
    )

    # Run and Assert
    ctgan = CTGAN(epochs=1)
    with pytest.raises(InvalidDataError, match=error_message):
        ctgan.fit(data, discrete_columns)


def test_synthesizer_sample():
    """Test the CTGAN samples the correct datatype."""
    data = pd.DataFrame({'discrete': np.random.choice(['a', 'b', 'c'], 100)})
    discrete_columns = ['discrete']

    ctgan = CTGAN(epochs=1)
    ctgan.fit(data, discrete_columns)

    samples = ctgan.sample(1000, 'discrete', 'a')
    assert isinstance(samples, pd.DataFrame)


def test_save_load():
    """Test the CTGAN load/save methods."""
    data = pd.DataFrame({
        'continuous': np.random.random(100),
        'discrete': np.random.choice(['a', 'b', 'c'], 100),
    })
    discrete_columns = ['discrete']

    ctgan = CTGAN(epochs=1)
    ctgan.fit(data, discrete_columns)

    with tf.TemporaryDirectory() as temporary_directory:
        ctgan.save(temporary_directory + 'test_tvae.pkl')
        ctgan = CTGAN.load(temporary_directory + 'test_tvae.pkl')

    sampled = ctgan.sample(1000)
    assert set(sampled.columns) == {'continuous', 'discrete'}
    assert set(sampled['discrete'].unique()) == {'a', 'b', 'c'}


def test_wrong_discrete_columns_dataframe():
    """Test the CTGAN correctly crashes when passed non-existing discrete columns."""
    data = pd.DataFrame({'discrete': ['a', 'b']})
    discrete_columns = ['b', 'c']

    ctgan = CTGAN(epochs=1)
    with pytest.raises(ValueError, match="Invalid columns found: {'.*', '.*'}"):
        ctgan.fit(data, discrete_columns)


def test_wrong_discrete_columns_numpy():
    """Test the CTGAN correctly crashes when passed non-existing discrete columns."""
    data = pd.DataFrame({'discrete': ['a', 'b']})
    discrete_columns = [0, 1]

    ctgan = CTGAN(epochs=1)
    with pytest.raises(ValueError, match=r'Invalid columns found: \[1\]'):
        ctgan.fit(data.to_numpy(), discrete_columns)


def test_wrong_sampling_conditions():
    """Test the CTGAN correctly crashes when passed incorrect sampling conditions."""
    data = pd.DataFrame({
        'continuous': np.random.random(100),
        'discrete': np.random.choice(['a', 'b', 'c'], 100),
    })
    discrete_columns = ['discrete']

    ctgan = CTGAN(epochs=1)
    ctgan.fit(data, discrete_columns)

    with pytest.raises(ValueError, match="The column_name `cardinal` doesn't exist in the data."):
        ctgan.sample(1, 'cardinal', "doesn't matter")

    with pytest.raises(ValueError):  # noqa: RDT currently incorrectly raises a tuple instead of a string
        ctgan.sample(1, 'discrete', 'd')


def test_fixed_random_seed():
    """Test the CTGAN with a fixed seed.

    Expect that when the random seed is reset with the same seed, the same sequence
    of data will be produced. Expect that the data generated with the seed is
    different than randomly sampled data.
    """
    # Setup
    data = pd.DataFrame({
        'continuous': np.random.random(100),
        'discrete': np.random.choice(['a', 'b', 'c'], 100),
    })
    discrete_columns = ['discrete']

    ctgan = CTGAN(epochs=1, enable_gpu=False)

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


def test_ctgan_save_and_load(tmpdir):
    """Test that the ``CTGAN`` model can be saved and loaded."""
    # Setup
    data = pd.DataFrame({
        'continuous': np.random.random(100),
        'discrete': np.random.choice(['a', 'b', 'c'], 100),
    })
    discrete_columns = [1]

    ctgan = CTGAN(epochs=1)
    ctgan.fit(data.to_numpy(), discrete_columns)
    ctgan.set_random_state(0)

    ctgan.sample(100)
    model_path = tmpdir / 'model.pkl'

    # Save
    ctgan.save(str(model_path))

    # Load
    loaded_instance = CTGAN.load(str(model_path))
    loaded_instance.sample(100)
