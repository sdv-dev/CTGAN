
"""BaseSynthesizer unit testing module."""

from unittest.mock import MagicMock, call, patch

from ctgan.synthesizers.base import BaseSynthesizer, random_state


@patch('ctgan.synthesizers.base.np.random')
def test_valid_random_state(random_mock):
    """Test the ``random_state`` attribute with a valid random state.

    Expect that the decorated function uses the random_state attribute.
    """
    # Setup
    my_function = MagicMock()
    instance = MagicMock()
    instance.random_state = 42

    args = {'some', 'args'}
    kwargs = {'keyword': 'value'}

    random_mock.get_state.return_value = 'random state'
    desired_state_mock = MagicMock()
    desired_state_mock.get_state.return_value = 'desired random state'
    random_mock.RandomState.return_value = desired_state_mock

    # Run
    decorated_function = random_state(my_function)
    decorated_function(instance, *args, **kwargs)

    # Assert
    my_function.assert_called_once_with(instance, *args, **kwargs)

    instance.assert_not_called
    random_mock.get_state.assert_called_once_with()
    random_mock.RandomState.assert_called_once_with(seed=42)
    random_mock.set_state.assert_has_calls(
        [call('desired random state'), call('random state')])
    assert random_mock.set_state.call_count == 2


@patch('ctgan.synthesizers.base.np.random')
def test_no_random_state(random_mock):
    """Test the ``random_state`` attribute with no random state.

    Expect that the decorated function calls the original function
    when there is no random state.
    """
    # Setup
    my_function = MagicMock()
    instance = MagicMock()
    instance.random_state = None

    args = {'some', 'args'}
    kwargs = {'keyword': 'value'}

    random_mock.get_state.return_value = 'random state'

    # Run
    decorated_function = random_state(my_function)
    decorated_function(instance, *args, **kwargs)

    # Assert
    my_function.assert_called_once_with(instance, *args, **kwargs)

    instance.assert_not_called
    random_mock.get_state.assert_not_called()
    random_mock.RandomState.assert_not_called()
    random_mock.set_state.assert_not_called()


class TestBaseSynthesizer:

    def test_set_random_state(self):
        """Test ``set_random_state`` works as expected."""
        # Setup
        instance = BaseSynthesizer()

        # Run
        instance.set_random_state(3)

        # Assert
        assert instance.random_state == 3
