"""BaseSynthesizer module."""

import contextlib

import numpy as np
import torch


@contextlib.contextmanager
def random_seed(seed):
    """Context manager for managing the random seed.

    Args:
        seed (int):
            The random seed.
    """
    state = np.random.get_state()
    torch_state = torch.get_rng_state()

    np.random.seed(seed)
    torch.manual_seed(seed)

    try:
        yield
    finally:
        np.random.set_state(state)
        torch.set_rng_state(torch_state)


def random_state(function):
    """Set the random state before calling the function.

    Args:
        function (Callable):
            The function to wrap around.
    """

    def wrapper(self, *args, **kwargs):
        if self._random_seed is None:
            return function(self, *args, **kwargs)

        else:
            with random_seed(self._random_seed):
                return function(self, *args, **kwargs)

    return wrapper


class BaseSynthesizer:
    """Base class for all default synthesizers of ``CTGAN``.

    This should contain the save/load methods.
    """

    _random_seed = None

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

    def set_random_seed(self, random_seed):
        """Set the random seed.

        Args:
            random_seed (int):
                Seed for the random generator.
        """
        self._random_seed = random_seed
