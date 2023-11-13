"""Data transformer intergration testing module."""

from unittest import TestCase

import numpy as np
import pandas as pd

from ctgan.data_transformer import DataTransformer


class TestDataTransformer(TestCase):

    def test_constant(self):
        """Test transforming a dataframe containing constant values."""
        # Setup
        data = pd.DataFrame({'cnt': [123] * 1000})
        transformer = DataTransformer()

        # Run
        transformer.fit(data, [])
        new_data = transformer.transform(data)
        transformer.inverse_transform(new_data)

        # Assert transformed values are between -1 and 1
        assert (new_data[:, 0] > -np.ones(len(new_data))).all()
        assert (new_data[:, 0] < np.ones(len(new_data))).all()

        # Assert transformed values are a gaussian centered in 0 and with std ~ 0
        assert -.1 < np.mean(new_data[:, 0]) < .1
        assert 0 <= np.std(new_data[:, 0]) < .1

        # Assert there are at most `max_columns=10` one hot columns
        assert new_data.shape[0] == 1000
        assert new_data.shape[1] <= 11
        assert np.isin(new_data[:, 1:], [0, 1]).all()

    def test_df_continuous(self):
        """Test transforming a dataframe containing only continuous values."""
        # Setup
        data = pd.DataFrame({'col': np.random.normal(size=1000)})
        transformer = DataTransformer()

        # Run
        transformer.fit(data, [])
        new_data = transformer.transform(data)
        transformer.inverse_transform(new_data)

        # Assert transformed values are between -1 and 1
        assert (new_data[:, 0] > -np.ones(len(new_data))).all()
        assert (new_data[:, 0] < np.ones(len(new_data))).all()

        # Assert transformed values are a gaussian centered in 0 and with std = 1/4
        assert -.1 < np.mean(new_data[:, 0]) < .1
        assert .2 < np.std(new_data[:, 0]) < .3

        # Assert there are at most `max_columns=10` one hot columns
        assert new_data.shape[0] == 1000
        assert new_data.shape[1] <= 11
        assert np.isin(new_data[:, 1:], [0, 1]).all()

    def test_df_categorical_constant(self):
        """Test transforming a dataframe containing only constant categorical values."""
        # Setup
        data = pd.DataFrame({'cnt': [123] * 1000})
        transformer = DataTransformer()

        # Run
        transformer.fit(data, ['cnt'])
        new_data = transformer.transform(data)
        transformer.inverse_transform(new_data)

        # Assert there is only 1 one hot vector
        assert np.array_equal(new_data, np.ones((len(data), 1)))

    def test_df_categorical(self):
        """Test transforming a dataframe containing only categorical values."""
        # Setup
        data = pd.DataFrame({'cat': np.random.choice(['a', 'b', 'c'], size=1000)})
        transformer = DataTransformer()

        # Run
        transformer.fit(data, ['cat'])
        new_data = transformer.transform(data)
        transformer.inverse_transform(new_data)

        # Assert there are 3 one hot vectors
        assert new_data.shape[0] == 1000
        assert new_data.shape[1] == 3
        assert np.isin(new_data[:, 1:], [0, 1]).all()

    def test_df_mixed(self):
        """Test transforming a dataframe containing mixed data types."""
        # Setup
        data = pd.DataFrame({
            'num': np.random.normal(size=1000),
            'cat': np.random.choice(['a', 'b', 'c'], size=1000)
        })
        transformer = DataTransformer()

        # Run
        transformer.fit(data, ['cat'])
        new_data = transformer.transform(data)
        transformer.inverse_transform(new_data)

        # Assert transformed numerical values are between -1 and 1
        assert (new_data[:, 0] > -np.ones(len(new_data))).all()
        assert (new_data[:, 0] < np.ones(len(new_data))).all()

        # Assert transformed numerical values are a gaussian centered in 0 and with std = 1/4
        assert -.1 < np.mean(new_data[:, 0]) < .1
        assert .2 < np.std(new_data[:, 0]) < .3

        # Assert there are at most `max_columns=10` one hot columns for the numerical values
        # and 3 for the categorical ones
        assert new_data.shape[0] == 1000
        assert 5 <= new_data.shape[1] <= 17
        assert np.isin(new_data[:, 1:], [0, 1]).all()

    def test_numpy(self):
        """Test transforming a numpy array."""
        # Setup
        data = pd.DataFrame({
            'num': np.random.normal(size=1000),
            'cat': np.random.choice(['a', 'b', 'c'], size=1000)
        })
        data = np.array(data)
        transformer = DataTransformer()

        # Run
        transformer.fit(data, [1])
        new_data = transformer.transform(data)
        transformer.inverse_transform(new_data)

        # Assert transformed numerical values are between -1 and 1
        assert (new_data[:, 0] > -np.ones(len(new_data))).all()
        assert (new_data[:, 0] < np.ones(len(new_data))).all()

        # Assert transformed numerical values are a gaussian centered in 0 and with std = 1/4
        assert -.1 < np.mean(new_data[:, 0]) < .1
        assert .2 < np.std(new_data[:, 0]) < .3

        # Assert there are at most `max_columns=10` one hot columns for the numerical values
        # and 3 for the categorical ones
        assert new_data.shape[0] == 1000
        assert 5 <= new_data.shape[1] <= 17
        assert np.isin(new_data[:, 1:], [0, 1]).all()
