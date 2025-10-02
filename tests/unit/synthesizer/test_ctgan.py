"""CTGAN unit testing module."""

from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from ctgan.data_transformer import SpanInfo
from ctgan.errors import InvalidDataError
from ctgan.synthesizers.ctgan import CTGAN, Discriminator, Generator, Residual


class TestDiscriminator(TestCase):
    def test___init__(self):
        """Test `__init__` for a generic case.

        Make sure 'self.seq' has same length as 3*`discriminator_dim` + 1.

        Setup:
            - Create Discriminator

        Input:
            - input_dim = positive integer
            - discriminator_dim = list of integers
            - pack = positive integer

        Output:
            - None

        Side Effects:
            - Set `self.seq`, `self.pack` and `self.packdim`
        """
        discriminator_dim = [1, 2, 3]
        discriminator = Discriminator(input_dim=50, discriminator_dim=discriminator_dim, pac=7)

        assert discriminator.pac == 7
        assert discriminator.pacdim == 350
        assert len(discriminator.seq) == 3 * len(discriminator_dim) + 1

    def test_forward(self):
        """Test `test_forward` for a generic case.

        Check that the output shapes are correct.
        We can also test that all parameters have a gradient attached to them
        by running `encoder.parameters()`. To do that, we just need to use `loss.backward()`
        for some loss, like `loss = torch.mean(output)`. Notice that the input_dim = input_size.

        Setup:
            - initialize with input_size, discriminator_dim, pac
            - Create random tensor as input

        Input:
            - input = random tensor of shape (N, input_size)

        Output:
            - tensor of shape (N/pac, 1)
        """
        discriminator = Discriminator(input_dim=50, discriminator_dim=[100, 200, 300], pac=7)
        output = discriminator(torch.randn(70, 50))
        assert output.shape == (10, 1)

        # Check to make sure no gradients attached
        for parameter in discriminator.parameters():
            assert parameter.grad is None

        # Backpropagate
        output.mean().backward()

        # Check to make sure all parameters have gradients
        for parameter in discriminator.parameters():
            assert parameter.grad is not None


class TestResidual(TestCase):
    def test_forward(self):
        """Test `test_forward` for a generic case.

        Check that the output shapes are correct.
        We can also test that all parameters have a gradient attached to them
        by running `encoder.parameters()`. To do that, we just need to use `loss.backward()`
        for some loss, like `loss = torch.mean(output)`.

        Setup:
            - initialize with input_size, output_size
            - Create random tensor as input

        Input:
            - input = random tensor of shape (N, input_size)

        Output:
            - tensor of shape (N, input_size + output_size)
        """
        residual = Residual(10, 2)
        output = residual(torch.randn(100, 10))
        assert output.shape == (100, 12)

        # Check to make sure no gradients attached
        for parameter in residual.parameters():
            assert parameter.grad is None

        # Backpropagate
        output.mean().backward()

        # Check to make sure all parameters have gradients
        for parameter in residual.parameters():
            assert parameter.grad is not None


class TestGenerator(TestCase):
    def test___init__(self):
        """Test `__init__` for a generic case.

        Make sure `self.seq` has same length as `generator_dim` + 1.

        Setup:
            - Create Generator

        Input:
            - embedding_dim = positive integer
            - generator_dim = list of integers
            - data_dim = positive integer

        Output:
            - None

        Side Effects:
            - Set `self.seq`
        """
        generator_dim = [1, 2, 3]
        generator = Generator(embedding_dim=50, generator_dim=generator_dim, data_dim=7)

        assert len(generator.seq) == len(generator_dim) + 1

    def test_forward(self):
        """Test `test_forward` for a generic case.

        Check that the output shapes are correct.
        We can also test that all parameters have a gradient attached to them
        by running `encoder.parameters()`. To do that, we just need to use `loss.backward()`
        for some loss, like `loss = torch.mean(output)`.

        Setup:
            - initialize with embedding_dim, generator_dim, data_dim
            - Create random tensor as input

        Input:
            - input = random tensor of shape (N, input_size)

        Output:
            - tensor of shape (N, data_dim)
        """
        generator = Generator(embedding_dim=60, generator_dim=[100, 200, 300], data_dim=500)
        output = generator(torch.randn(70, 60))
        assert output.shape == (70, 500)

        # Check to make sure no gradients attached
        for parameter in generator.parameters():
            assert parameter.grad is None

        # Backpropagate
        output.mean().backward()

        # Check to make sure all parameters have gradients
        for parameter in generator.parameters():
            assert parameter.grad is not None


def _assert_is_between(data, lower, upper):
    """Assert all values of the tensor 'data' are within range."""
    assert all((data >= lower).numpy().tolist())
    assert all((data <= upper).numpy().tolist())


class TestCTGAN(TestCase):
    @patch('ctgan.synthesizers.ctgan.validate_and_set_device')
    def test___init__(self, mock_validate_and_set_device):
        """Test the `__init__` method."""
        # Setup
        mock_validate_and_set_device.return_value = 'cpu'

        # Run
        synth = CTGAN()

        # Assert
        assert synth._embedding_dim == 128
        assert synth._generator_dim == (256, 256)
        assert synth._discriminator_dim == (256, 256)
        assert synth._batch_size == 500
        assert synth._epochs == 300
        assert synth.pac == 10
        assert synth.loss_values is None
        assert synth._generator is None
        assert synth._data_sampler is None
        assert synth._verbose is False
        assert synth._enable_gpu is True
        assert synth._device == 'cpu'
        mock_validate_and_set_device.assert_called_once_with(True, None)

    def test__apply_activate_(self):
        """Test `_apply_activate` for tables with both continuous and categoricals.

        Check every continuous column has all values between -1 and 1
        (since they are normalized), and check every categorical column adds up to 1.

        Setup:
            - Mock `self._transformer.output_info_list`

        Input:
            - data = tensor of shape (N, data_dims)

        Output:
            - tensor = tensor of shape (N, data_dims)
        """
        model = CTGAN()
        model._transformer = Mock()
        model._transformer.output_info_list = [
            [SpanInfo(3, 'softmax')],
            [SpanInfo(1, 'tanh'), SpanInfo(2, 'softmax')],
        ]

        data = torch.randn(100, 6)
        result = model._apply_activate(data)

        assert result.shape == (100, 6)
        _assert_is_between(result[:, 0:3], 0.0, 1.0)
        _assert_is_between(result[:3], -1.0, 1.0)
        _assert_is_between(result[:, 4:6], 0.0, 1.0)

    def test__cond_loss(self):
        """Test `_cond_loss`.

        Test that the loss is purely a function of the target categorical.

        Setup:
            - mock transformer.output_info_list
            - create two categoricals, one continuous
            - compute the conditional loss, conditioned on the 1st categorical
            - compare the loss to the cross-entropy of the 1st categorical, manually computed

        Input:
            data - the synthetic data generated by the model
            c - a tensor with the same shape as the data but with only a specific one-hot vector
                corresponding to the target column filled in
            m - binary mask used to select the categorical column to condition on

        Output:
            loss scalar; this should only be affected by the target column

        Note:
            - even though the implementation of this is probably right, I'm not sure if the idea
              behind it is correct
        """
        model = CTGAN()
        model._transformer = Mock()
        model._transformer.output_info_list = [
            [SpanInfo(1, 'tanh'), SpanInfo(2, 'softmax')],
            [SpanInfo(3, 'softmax')],  # this is the categorical column we are conditioning on
            [SpanInfo(2, 'softmax')],  # this is the categorical column we are bry jrbec on
        ]

        data = torch.tensor([
            # first 3 dims ignored, next 3 dims are the prediction, last 2 dims are ignored
            [0.0, -1.0, 0.0, 0.05, 0.05, 0.9, 0.1, 0.4],
        ])

        c = torch.tensor([
            # first 3 dims are a one-hot for the categorical,
            # next 2 are for a different categorical that we are not conditioning on
            # (continuous values are not stored in this tensor)
            [0.0, 0.0, 1.0, 0.0, 0.0],
        ])

        # this indicates that we are conditioning on the first categorical
        m = torch.tensor([[1, 0]])

        result = model._cond_loss(data, c, m)
        expected = torch.nn.functional.cross_entropy(
            torch.tensor([
                [0.05, 0.05, 0.9],  # 3 categories, one hot
            ]),
            torch.tensor([2]),
        )

        assert (result - expected).abs() < 1e-3

    def test__validate_discrete_columns(self):
        """Test `_validate_discrete_columns` if the discrete column doesn't exist.

        Check the appropriate error is raised if `discrete_columns` is invalid, both
        for numpy arrays and dataframes.

        Setup:
            - Create dataframe with a discrete column
            - Define `discrete_columns` as something not in the dataframe

        Input:
            - train_data = 2-dimensional numpy array or a pandas.DataFrame
            - discrete_columns = list of strings or integers

        Output:
            None

        Side Effects:
            - Raises error if the discrete column is invalid.

        Note:
            - could create another function for numpy array
        """
        data = pd.DataFrame({'discrete': ['a', 'b']})
        discrete_columns = ['doesnt exist']

        ctgan = CTGAN(epochs=1)
        with pytest.raises(ValueError, match=r'Invalid columns found: {\'doesnt exist\'}'):
            ctgan.fit(data, discrete_columns)

    def test__validate_null_data(self):
        """Test `_validate_null_data` with pandas and numpy data.

        Check the appropriate error is raised if null values are present in
        continuous columns, both for numpy arrays and dataframes.
        """
        # Setup
        discrete_df = pd.DataFrame({'discrete': ['a', 'b']})
        discrete_array = np.array([['a'], ['b']])
        continuous_no_nulls_df = pd.DataFrame({'continuous': [0, 1]})
        continuous_no_nulls_array = np.array([[0], [1]])
        continuous_with_null_df = pd.DataFrame({'continuous': [1, np.nan]})
        continuous_with_null_array = np.array([[1], [np.nan]])
        ctgan = CTGAN(epochs=1)
        error_message = (
            'CTGAN does not support null values in the continuous training data. '
            'Please remove all null values from your continuous training data.'
        )

        # Test discrete DataFrame fits without error
        ctgan.fit(discrete_df, ['discrete'])

        # Test discrete array fits without error
        ctgan.fit(discrete_array, [0])

        # Test continuous DataFrame without nulls fits without error
        ctgan.fit(continuous_no_nulls_df)

        # Test continuous array without nulls fits without error
        ctgan.fit(continuous_no_nulls_array)

        # Test nulls in continuous columns DataFrame errors on fit
        with pytest.raises(InvalidDataError, match=error_message):
            ctgan.fit(continuous_with_null_df)

        # Test nulls in continuous columns array errors on fit
        with pytest.raises(InvalidDataError, match=error_message):
            ctgan.fit(continuous_with_null_array)
