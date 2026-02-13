import platform
import warnings

import torch


def get_enable_gpu_value(enable_gpu, cuda):
    """Validate both the `enable_gpu` and `cuda` parameters.

    The logic here is to:
    - Raise a warning if `cuda` is set because it's deprecated.
    - Raise an error if both parameters are set in a conflicting way.
    - Return the resolved `enable_gpu` value.
    """
    if cuda is not None:
        warnings.warn(
            '`cuda` parameter is deprecated and will be removed in a future release. '
            'Please use `enable_gpu` instead.',
            FutureWarning,
        )
        if not enable_gpu:
            raise ValueError(
                'Cannot resolve the provided values of `cuda` and `enable_gpu` parameters. '
                'Please use only `enable_gpu`.'
            )

        enable_gpu = cuda

    return enable_gpu


def _set_device(enable_gpu, device=None):
    """Set the torch device based on the `enable_gpu` parameter and system capabilities."""
    if device:
        return torch.device(device)

    if enable_gpu:
        if platform.system() == 'Darwin':  # macOS
            if (
                platform.machine() == 'arm64'
                and getattr(torch.backends, 'mps', None)
                and torch.backends.mps.is_available()
            ):
                device = 'mps'
            else:
                device = 'cpu'
        else:  # Linux/Windows
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = 'cpu'

    return torch.device(device)


def validate_and_set_device(enable_gpu, cuda):
    enable_gpu = get_enable_gpu_value(enable_gpu, cuda)
    return _set_device(enable_gpu)


def _format_score(score):
    """Format a score as a fixed-length string ``±XX.XX``.

    Values are clipped to the range ``[-99.99, +99.99]`` so the result
    is always exactly 6 characters.
    """
    score = max(-99.99, min(99.99, score))
    sign = '+' if score >= 0 else '-'
    return f'{sign}{abs(score):05.2f}'
