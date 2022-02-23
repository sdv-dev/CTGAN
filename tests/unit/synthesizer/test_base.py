
"""BaseSynthesizer unit testing module."""

from unittest.mock import MagicMock, call, patch

import numpy as np
import torch

from ctgan.synthesizers.base import BaseSynthesizer, random_state


@patch('ctgan.synthesizers.base.torch')
@patch('ctgan.synthesizers.base.np.random')
def test_valid_random_state(random_mock, torch_mock):
    """Test the ``random_state`` attribute with a valid random state.

    Expect that the decorated function uses the random_state attribute.
    """
    # Setup
    my_function = MagicMock()
    instance = MagicMock()

    random_state_mock = MagicMock()
    random_state_mock.get_state.return_value = 'desired numpy state'
    torch_generator_mock = MagicMock()
    torch_generator_mock.get_state.return_value = 'desired torch state'
    instance.random_states = (random_state_mock, torch_generator_mock)

    args = {'some', 'args'}
    kwargs = {'keyword': 'value'}

    random_mock.RandomState.return_value = random_state_mock
    random_mock.get_state.return_value = 'random state'
    torch_mock.Generator.return_value = torch_generator_mock
    torch_mock.get_rng_state.return_value = 'torch random state'

    # Run
    decorated_function = random_state(my_function)
    decorated_function(instance, *args, **kwargs)

    # Assert
    my_function.assert_called_once_with(instance, *args, **kwargs)

    instance.assert_not_called
    assert random_mock.get_state.call_count == 2
    assert torch_mock.get_rng_state.call_count == 2
    random_mock.RandomState.assert_has_calls(
        [call().get_state(), call(), call().set_state('random state')])
    random_mock.set_state.assert_has_calls([call('desired numpy state'), call('random state')])
    torch_mock.set_rng_state.assert_has_calls(
        [call('desired torch state'), call('torch random state')])


@patch('ctgan.synthesizers.base.torch')
@patch('ctgan.synthesizers.base.np.random')
def test_no_random_seed(random_mock, torch_mock):
    """Test the ``random_state`` attribute with no random state.

    Expect that the decorated function calls the original function
    when there is no random state.
    """
    # Setup
    my_function = MagicMock()
    instance = MagicMock()
    instance.random_states = None

    args = {'some', 'args'}
    kwargs = {'keyword': 'value'}

    # Run
    decorated_function = random_state(my_function)
    decorated_function(instance, *args, **kwargs)

    # Assert
    my_function.assert_called_once_with(instance, *args, **kwargs)

    instance.assert_not_called
    random_mock.get_state.assert_not_called()
    random_mock.RandomState.assert_not_called()
    random_mock.set_state.assert_not_called()
    torch_mock.get_rng_state.assert_not_called()
    torch_mock.Generator.assert_not_called()
    torch_mock.set_rng_state.assert_not_called()


class TestBaseSynthesizer:

    def test_set_random_state(self):
        """Test ``set_random_state`` works as expected."""
        # Setup
        instance = BaseSynthesizer()

        # Run
        instance.set_random_state(3)

        # Assert
        assert isinstance(instance.random_states, tuple)
        assert isinstance(instance.random_states[0], np.random.RandomState)
        assert isinstance(instance.random_states[1], torch.Generator)

    def test_set_random_state_with_none(self):
        """Test ``set_random_state`` with None."""
        # Setup
        instance = BaseSynthesizer()

        # Run and assert
        instance.set_random_state(3)
        assert instance.random_states is not None

        instance.set_random_state(None)
        assert instance.random_states is None
