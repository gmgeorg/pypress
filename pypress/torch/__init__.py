"""PyTorch implementation of PRESS.

Needs PyTorch, which is an optional dependency of `pypress`:
`pip install "pypress[torch] @ git+https://github.com/gmgeorg/pypress.git"`.

Modules: `layers` (`PredictiveStateSimplex`, `PredictiveStateMeans`,
`PredictiveStateParams`, `PRESS`), `regularizers`, `statistics`, `schedules` and
`activations`. `regularization_loss` sums the penalties of activity regularizers
attached to layers.
"""

try:
    import torch  # noqa: F401
except ImportError as error:  # pragma: no cover - exercised in a subprocess
    raise ImportError(
        "pypress.torch needs PyTorch, which is not installed. Install it with: "
        "pip install 'pypress[torch] @ git+https://github.com/gmgeorg/pypress.git'"
    ) from error

from .layers import regularization_loss  # noqa: E402

__all__ = ["regularization_loss"]
