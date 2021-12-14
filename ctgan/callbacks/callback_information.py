from dataclasses import dataclass


@dataclass
class CallbackInformation:
    """
    Class that holds information about the callback.
    """
    early_stop: bool = False
