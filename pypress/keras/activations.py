"""Activation functions and their inverses for PRESS layers."""

from typing import Callable

import tensorflow as tf


ACTIVATION_INVERSES = {
    "linear": lambda x: x,
    "sigmoid": lambda x: tf.math.log(x / (1 - x)),
    # numerically stable: for x > 20 softplus is essentially linear
    "softplus": lambda x: tf.where(x > 20.0, x, tf.math.log(tf.math.expm1(x))),
    "tanh": lambda x: tf.math.atanh(x),
    "exponential": lambda x: tf.math.log(x + 1e-10),
    # Leaky ReLU: Inverse is x/alpha for negative values
    # Assuming default alpha=0.2; adjust as needed
    "leaky_relu": lambda x, alpha=0.2: tf.where(x >= 0, x, x / alpha),
    # ELU: Inverse is log(x/alpha + 1) for negative values
    "elu": lambda x, alpha=1.0: tf.where(x >= 0, x, tf.math.log(x / alpha + 1.0)),
    # Softsign: x / (1 + |x|) -> inverse is x / (1 - |x|)
    "softsign": lambda x: x / (1.0 - tf.math.abs(x)),
    # Softmax: No unique inverse, but log(y) provides the logits
    # shifted by a constant (Log-Softmax identity)
    "softmax": lambda x: tf.math.log(x + 1e-10),
    "selu": lambda x, alpha=1.67326, scale=1.0507: tf.where(
        x >= 0, x / scale, tf.math.log(x / (scale * alpha) + 1.0)
    ),
}


def get_inverse_activation(activation: str) -> Callable:
    """Get the inverse activation function.

    This function returns the mathematical inverse of common activation functions,
    which is useful for converting values from the original scale (after activation)
    back to logits (before activation).

    Args:
        activation: Name of the activation function. Supported activations:
            - "linear": identity function (f^-1(x) = x)
            - "sigmoid": logit function (f^-1(p) = log(p/(1-p)))
            - "softplus": inverse softplus (f^-1(x) = log(exp(x) - 1))
            - "tanh": inverse tanh (f^-1(x) = atanh(x))
            - "exponential": natural logarithm (f^-1(x) = log(x))

    Returns:
        Callable that computes the inverse activation function.

    Raises:
        ValueError: If activation is not recognized or has no unique inverse.
            Activations without inverses include "relu" and "softmax".

    Example:
        >>> # Convert probability to logit
        >>> inverse_sigmoid = get_inverse_activation("sigmoid")
        >>> prob = tf.constant(0.8)
        >>> logit = inverse_sigmoid(prob)
        >>> # Verify: sigmoid(logit) ≈ 0.8
        >>> tf.nn.sigmoid(logit)
        <tf.Tensor: ... numpy=0.8>
    """
    if activation not in ACTIVATION_INVERSES:
        raise ValueError(
            f"Unknown activation '{activation}'. Supported activations: "
            f"{list(ACTIVATION_INVERSES.keys())}. For custom activations, "
            f"please use a custom initializer."
        )

    inverse_fn = ACTIVATION_INVERSES[activation]
    if inverse_fn is None:
        raise ValueError(
            f"Activation '{activation}' has no unique inverse. "
            f"Cannot automatically convert from original scale. "
            f"Please use a custom initializer or a different activation."
        )

    return inverse_fn
