"""BaseSynthesizer module."""

import contextlib

import numpy as np
import torch


@contextlib.contextmanager
def set_random_state(random_state, torch_random_state, set_model_random_state,
                     set_model_torch_random_state):
    """Context manager for managing the random state.

    Args:
        random_state (int or np.random.RandomState):
            The random seed or RandomState.
        set_model_random_state (function):
            Function to set the random state on the model.
    """
    original_state = np.random.get_state()
    original_torch_state = torch.get_rng_state()

    if isinstance(random_state, int):
        random_state = np.random.RandomState(seed=random_state).get_state()
    elif isinstance(random_state, np.random.RandomState):
        random_state = random_state.get_state()
    elif not isinstance(random_state, tuple):
        raise TypeError(f'RandomState {random_state} is an unexpected type. '
                        'Expected to be int, np.random.RandomState, or tuple.')

    if not torch_random_state:
        torch_random_state = torch.from_numpy(np.asarray(random_state))

    torch.set_rng_state(torch_random_state)
    np.random.set_state(random_state)

    try:
        yield
    finally:
        set_model_random_state(np.random.get_state())
        set_model_torch_random_state(torch.get_rng_state())
        np.random.set_state(original_state)
        torch.set_rng_state(original_torch_state)


def random_state(function):
    """Set the random state before calling the function.

    Args:
        function (Callable):
            The function to wrap around.
    """

    def wrapper(self, *args, **kwargs):
        if self.random_state is None:
            return function(self, *args, **kwargs)
        else:
            with set_random_state(
                    self._random_state,
                    self._torch_random_state,
                    self.set_random_state,
                    self._set_torch_random_state):
                return function(self, *args, **kwargs)

    return wrapper


class BaseSynthesizer:
    """Base class for all default synthesizers of ``CTGAN``.

    This should contain the save/load methods.
    """

    _random_state = None
    _torch_random_state = None

    def save(self, path):
        """Save the model in the passed `path`."""
        device_backup = self._device
        self.set_device(torch.device('cpu'))
        torch.save(self, path)
        self.set_device(device_backup)

    @classmethod
    def load(cls, path):
        """Load the model stored in the passed `path`."""
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = torch.load(path)
        model.set_device(device)
        return model

    def set_random_state(self, random_state):
        """Set the random state.

        Args:
            random_state (int, np.random.RandomState, or None):
                Seed or RandomState for the random generator.
        """
        self._random_state = random_state

    def _set_torch_random_state(self, torch_random_state):
        """Set the torch random state.

        The torch random state is initially based off of the ``self.random_state``.
        Afterwards, we track it with ``self.torch_random_state``.

        Args:
            torch_random_state (torch.ByteTensor):
                The torch random state.
        """
        self._torch_random_state = torch_random_state
