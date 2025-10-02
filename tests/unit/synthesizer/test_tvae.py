"""TVAE unit testing module."""

from unittest.mock import MagicMock, Mock, call, patch

import pandas as pd

from ctgan.synthesizers import TVAE


class TestTVAE:
    @patch('ctgan.synthesizers.tvae.validate_and_set_device')
    def test___init__(self, mock_validate_and_set_device):
        """Test the `__init__` method."""
        # Setup
        mock_validate_and_set_device.return_value = 'cpu'

        # Run
        synth = TVAE()

        # Assert
        assert synth.embedding_dim == 128
        assert synth.compress_dims == (128, 128)
        assert synth.decompress_dims == (128, 128)
        assert synth.batch_size == 500
        assert synth.epochs == 300
        assert synth.loss_values.equals(pd.DataFrame(columns=['Epoch', 'Batch', 'Loss']))
        assert synth.verbose is False
        assert synth._enable_gpu is True
        assert synth._device == 'cpu'
        mock_validate_and_set_device.assert_called_once_with(True, None)

    @patch('ctgan.synthesizers.tvae._loss_function')
    @patch('ctgan.synthesizers.tvae.tqdm')
    def test_fit_verbose(self, tqdm_mock, loss_func_mock):
        """Test verbose parameter prints progress bar."""
        # Setup
        epochs = 1

        def mock_iter():
            for i in range(epochs):
                yield i

        def mock_add(a, b):
            mock_loss = Mock()
            mock_loss.detach().cpu().item.return_value = 1.23456789
            return mock_loss

        loss_mock = MagicMock()
        loss_mock.__add__ = mock_add
        loss_func_mock.return_value = (loss_mock, loss_mock)

        iterator_mock = MagicMock()
        iterator_mock.__iter__.side_effect = mock_iter
        tqdm_mock.return_value = iterator_mock
        synth = TVAE(epochs=epochs, verbose=True)
        train_data = pd.DataFrame({'col1': [0, 1, 2, 3, 4], 'col2': [10, 11, 12, 13, 14]})

        # Run
        synth.fit(train_data)

        # Assert
        tqdm_mock.assert_called_once_with(range(epochs), disable=False)
        assert iterator_mock.set_description.call_args_list[0] == call('Loss: 0.000')
        assert iterator_mock.set_description.call_args_list[1] == call('Loss: 1.235')
        assert iterator_mock.set_description.call_count == 2
