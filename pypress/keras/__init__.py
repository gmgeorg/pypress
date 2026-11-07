"""Keras (TensorFlow) implementation of PRESS.

Needs TensorFlow, which is an optional dependency of `pypress`:
`pip install "pypress[tf] @ git+https://github.com/gmgeorg/pypress.git"`.
"""

try:
    import tensorflow  # noqa: F401
except ImportError as error:  # pragma: no cover - exercised in a subprocess
    raise ImportError(
        "pypress.keras needs TensorFlow, which is not installed. Install it with: "
        "pip install 'pypress[tf] @ git+https://github.com/gmgeorg/pypress.git'"
    ) from error
