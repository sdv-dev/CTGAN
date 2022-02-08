"""BaseSynthesizer module."""

import contextlib

import numpy as np
import torch


@contextlib.contextmanager
def set_random_state(random_state, set_model_random_state):
    """Context manager for managing the random state.

    Args:
        random_state(int or np.random.RandomState):
            The random seed or RandomState.
        set_model_random_state (function):
            Function to set the random state on the model.
    """
    original_state = np.random.get_state()

    if isinstance(random_state, int):
        desired_state = np.random.RandomState(seed=random_state)
    else:
        desired_state = random_state

    np.random.set_state(desired_state.get_state())

    try:
        yield
    finally:
        set_model_random_state(desired_state)
        np.random.set_state(original_state)


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
            with set_random_state(self.random_state, self.set_random_state):
                return function(self, *args, **kwargs)

    return wrapper


class BaseSynthesizer:
    """Base class for all default synthesizers of ``CTGAN``.

    This should contain the save/load methods.
    """

    random_state = None

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
            random_state(int, np.random.RandomState, or None):
                Seed or RandomState for the random generator.
        """
        self.random_state = random_state
