from abc import ABC, abstractmethod

from ctgan.callbacks.callback_information import CallbackInformation


class Callback(ABC):

    @abstractmethod
    def callback(self, **kwargs) -> CallbackInformation:
        raise NotImplementedError("Callback must implement callback method.")
