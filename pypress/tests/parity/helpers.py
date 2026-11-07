"""Helpers shared by the parity tests: build a regularizer in either backend."""

import numpy as np
import tensorflow as tf
import torch

from pypress.keras import regularizers as keras_regularizers
from pypress.keras import schedules as keras_schedules
from pypress.torch import regularizers as torch_regularizers
from pypress.torch import schedules as torch_schedules

from .data import ATOL, RTOL


class Scheduled:
    """A `ScheduledValue` described by its arguments, built per backend."""

    def __init__(self, **kwargs):
        """Stores the `ScheduledValue` arguments."""
        self.kwargs = kwargs

    def build(self, backend):
        """The backend's `ScheduledValue`."""
        module = keras_schedules if backend == "keras" else torch_schedules
        return module.ScheduledValue(**self.kwargs)


# (class name, constructor kwargs): every hyperparameter setting the Keras tests use.
REGULARIZER_SETTINGS = [
    ("Uniform", dict(l2=0.5)),
    ("Uniform", dict(l2=1.5, ema_decay=0.8)),
    ("TargetEntropy", dict(l2=1.0)),
    ("TargetEntropy", dict(l2=2.0, target=0.7)),
    ("TargetEntropy", dict(l2=0.3, target=0.0)),
    ("TargetEntropy", dict(l2=1.0, entropy_fraction=0.3)),
    ("TargetEntropy", dict(l2=1.0, entropy_fraction=1.0, ema_decay=0.5)),
    ("StateSizeEntropy", dict(l2=1.0)),
    ("StateSizeEntropy", dict(l2=2.0, active_fraction=0.5)),
    ("StateSizeEntropy", dict(l2=1.0, active_fraction=0.7, ema_decay=0.9)),
    ("MinStateSize", dict(l2=1.0)),
    ("MinStateSize", dict(l2=3.0, min_share=0.05)),
    ("MinStateSize", dict(l2=1.0, min_share=0.1, ema_decay=0.9)),
    ("DegreesOfFreedom", dict(l2=1.0)),
    ("DegreesOfFreedom", dict(l2=0.5, target=3.0)),
    ("DegreesOfFreedom", dict(l2=0.5, target=3.0, normalize=True)),
    ("DegreesOfFreedom", dict(l2=2.0, target=2.0, ema_decay=0.9)),
    ("DegreesOfFreedom", dict(l2=2.0, target=2.0, normalize=True, ema_decay=0.5)),
    (
        "DegreesOfFreedom",
        dict(
            l2=Scheduled(start=0.0, end=2.0, duration_epochs=4),
            target=Scheduled(start=4.0, end=2.0, duration_epochs=4, schedule="cosine"),
        ),
    ),
    (
        "TargetEntropy",
        dict(
            l2=Scheduled(start=0.5, end=2.0, duration_epochs=3, schedule="exponential"),
            entropy_fraction=Scheduled(start=1.0, end=0.4, duration_epochs=3),
        ),
    ),
    (
        "MinStateSize",
        dict(
            l2=Scheduled(start=2.0, end=0.5, duration_epochs=5, start_epoch=1),
            min_share=Scheduled(start=0.02, end=0.1, duration_epochs=5),
        ),
    ),
]


def setting_id(setting):
    """A readable pytest id for a (class name, kwargs) setting."""
    name, kwargs = setting
    return (
        name
        + "-"
        + "-".join(
            f"{key}={'sched' if isinstance(value, Scheduled) else value}"
            for key, value in kwargs.items()
        )
    )


def build(backend, name, kwargs):
    """The regularizer called 'name' in 'backend' ("keras" or "torch")."""
    module = keras_regularizers if backend == "keras" else torch_regularizers
    resolved = {
        key: value.build(backend) if isinstance(value, Scheduled) else value
        for key, value in kwargs.items()
    }
    return getattr(module, name)(**resolved)


def keras_penalty(reg, weights):
    """(penalty value, gradient w.r.t. the weights) of a Keras regularizer."""
    x = tf.Variable(weights)
    with tf.GradientTape() as tape:
        penalty = reg.penalty(x)
    return penalty.numpy(), tape.gradient(penalty, x).numpy()


def torch_penalty(reg, weights):
    """(penalty value, gradient w.r.t. the weights) of a torch regularizer."""
    x = torch.tensor(weights, requires_grad=True)
    penalty = reg.penalty(x)
    penalty.backward()
    return penalty.detach().numpy(), x.grad.numpy()


def assert_penalties_equal(keras_reg, torch_reg, weights, message=""):
    """Value and gradient of both regularizers agree on 'weights'."""
    k_value, k_grad = keras_penalty(keras_reg, weights)
    t_value, t_grad = torch_penalty(torch_reg, weights)
    np.testing.assert_allclose(t_value, k_value, rtol=RTOL, atol=ATOL, err_msg=message)
    np.testing.assert_allclose(t_grad, k_grad, rtol=RTOL, atol=ATOL, err_msg=message)
