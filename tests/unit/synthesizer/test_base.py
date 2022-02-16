
"""BaseSynthesizer unit testing module."""

from unittest.mock import MagicMock, patch

from ctgan.synthesizers.base import BaseSynthesizer, random_state


@patch('ctgan.synthesizers.base.torch')
@patch('ctgan.synthesizers.base.np.random')
def test_valid_random_seed(random_mock, torch_mock):
    """Test the ``random_seed`` attribute with a valid random seed.

    Expect that the decorated function uses the random_state attribute.
    """
    # Setup
    my_function = MagicMock()
    instance = MagicMock()
    instance._random_seed = 42

    args = {'some', 'args'}
    kwargs = {'keyword': 'value'}

    random_mock.get_state.return_value = 'random state'
    torch_mock.get_rng_state.return_value = 'torch random state'

    # Run
    decorated_function = random_state(my_function)
    decorated_function(instance, *args, **kwargs)

    # Assert
    my_function.assert_called_once_with(instance, *args, **kwargs)

    instance.assert_not_called
    random_mock.get_state.assert_called_once_with()
    torch_mock.get_rng_state.assert_called_once_with()
    random_mock.seed.assert_called_once_with(42)
    random_mock.set_state.assert_called_once_with('random state')
    torch_mock.set_rng_state.assert_called_once_with('torch random state')


@patch('ctgan.synthesizers.base.torch')
@patch('ctgan.synthesizers.base.np.random')
def test_no_random_seed(random_mock, torch_mock):
    """Test the ``random_seed`` attribute with no random seed.

    Expect that the decorated function calls the original function
    when there is no random state.
    """
    # Setup
    my_function = MagicMock()
    instance = MagicMock()
    instance._random_seed = None

    args = {'some', 'args'}
    kwargs = {'keyword': 'value'}

    # Run
    decorated_function = random_state(my_function)
    decorated_function(instance, *args, **kwargs)

    # Assert
    my_function.assert_called_once_with(instance, *args, **kwargs)

    instance.assert_not_called
    random_mock.get_state.assert_not_called()
    random_mock.seed.assert_not_called()
    random_mock.set_state.assert_not_called()
    torch_mock.get_rng_state.assert_not_called()
    torch_mock.set_rng_state.assert_not_called()


class TestBaseSynthesizer:

    def test_set_random_seed(self):
        """Test ``set_random_seed`` works as expected."""
        # Setup
        instance = BaseSynthesizer()

        # Run
        instance.set_random_seed(3)

        # Assert
        assert instance._random_seed == 3
