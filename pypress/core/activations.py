"""Activation inverses for PRESS layers, in numpy.

Users give initial values on the original scale -- after the activation -- and
an initializer inverts the activation to get the logits the layer stores. The
inverses are framework-free, so Keras and PyTorch layers share them.
"""

from typing import Callable, Dict, List, Sequence, Union

import numpy as np


def safe_log(x: np.ndarray, neg_inf: float = -1e9) -> np.ndarray:
    """log(x) for x in [0, 1], with `neg_inf` substituted for exact zeros."""
    x = np.asarray(x, dtype=float)
    positive = x > 0
    return np.where(positive, np.log(np.where(positive, x, 1.0)), neg_inf)


ACTIVATION_INVERSES: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "linear": lambda x: x,
    "sigmoid": lambda x: np.log(x / (1 - x)),
    # numerically stable: for x > 20 softplus is essentially linear
    "softplus": lambda x: np.where(x > 20.0, x, np.log(np.expm1(x))),
    "tanh": np.arctanh,
    "exponential": safe_log,
    # Leaky ReLU: Inverse is x/alpha for negative values. Assumes alpha=0.2, the
    # default slope of the Keras activation.
    "leaky_relu": lambda x, alpha=0.2: np.where(x >= 0, x, x / alpha),
    # ELU: Inverse is log(x/alpha + 1) for negative values
    "elu": lambda x, alpha=1.0: np.where(x >= 0, x, np.log(x / alpha + 1.0)),
    # Softsign: x / (1 + |x|) -> inverse is x / (1 - |x|)
    "softsign": lambda x: x / (1.0 - np.abs(x)),
    # Softmax has no unique inverse (shift-invariant), but log(y) recovers the
    # logits up to that additive constant (Log-Softmax identity).
    "softmax": safe_log,
    "selu": lambda x, alpha=1.67326, scale=1.0507: np.where(
        x >= 0, x / scale, np.log(x / (scale * alpha) + 1.0)
    ),
}


def get_inverse_activation(activation: str) -> Callable[[np.ndarray], np.ndarray]:
    """The numpy inverse of the activation called 'activation'.

    Converts values from the original scale (after the activation) back to
    logits (before it).

    Args:
        activation: Name of the activation function; one of
            `ACTIVATION_INVERSES`.

    Raises:
        ValueError: If activation is not recognized, e.g. "relu" (not
            invertible). "softmax" is shift-invariant rather than uniquely
            invertible, but log(x) still recovers usable logits and is
            supported.
    """
    if activation not in ACTIVATION_INVERSES:
        raise ValueError(
            f"Unknown activation '{activation}'. Supported activations: "
            f"{list(ACTIVATION_INVERSES.keys())}. For custom activations, "
            f"please use a custom initializer."
        )
    return ACTIVATION_INVERSES[activation]


def invert(values: np.ndarray, activations: Union[str, Sequence[str]]) -> np.ndarray:
    """Logits whose activations equal 'values', column by column.

    Args:
      values: Original-scale values, shape (n_states, n_columns).
      activations: One activation name for all columns, or one per column.

    Raises:
      ValueError: on an activation without a known inverse, or when the number
        of activations does not match the number of columns.
    """
    values = np.asarray(values, dtype=float)
    n_columns = values.shape[1]
    names: List[str] = (
        [activations] * n_columns if isinstance(activations, str) else list(activations)
    )
    if len(names) != n_columns:
        raise ValueError(
            f"Need one activation per column ({n_columns}). Got {len(names)}."
        )
    with np.errstate(all="ignore"):
        columns = [
            np.asarray(get_inverse_activation(name)(values[:, i]), dtype=float)
            for i, name in enumerate(names)
        ]
    return np.stack(columns, axis=1)
