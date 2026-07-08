"""Shared device resolution for all models.

Preference order: CUDA, then MPS (Apple silicon), then CPU. Modules take an
optional ``device`` argument and fall back to this when it is None.
"""

import torch


def default_device() -> torch.device:
    """Best available device, in preference order CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
