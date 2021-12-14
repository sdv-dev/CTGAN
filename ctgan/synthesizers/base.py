"""BaseSynthesizer module."""


import torch

from ctgan.callbacks.callback_information import CallbackInformation


class BaseSynthesizer:
    """Base class for all default synthesizers of ``CTGAN``.

    This should contain the save/load methods.
    """

    def __init__(self):
        self.callback_info: CallbackInformation = None

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
