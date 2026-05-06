from __future__ import annotations

import importlib.util


def detect_acceleration_backend() -> str:
    """Return the best available backend for future ML workloads."""

    if importlib.util.find_spec("torch") is None:
        return "cpu"

    import torch

    if torch.cuda.is_available():
        return "cuda"

    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"

    return "cpu"
