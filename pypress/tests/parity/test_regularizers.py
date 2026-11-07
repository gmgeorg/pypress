"""The torch regularizers equal the Keras ones: values, gradients and the EMA."""

import numpy as np
import pytest
import tensorflow as tf
import torch

from pypress.keras import regularizers as keras_regularizers
from pypress.keras import schedules as keras_schedules
from pypress.torch import schedules as torch_schedules

from . import helpers
from .data import ATOL, BATCHES, RTOL, random_weights
from .helpers import REGULARIZER_SETTINGS, assert_penalties_equal, build, setting_id

_SETTINGS = pytest.mark.parametrize(
    "setting", REGULARIZER_SETTINGS, ids=[setting_id(s) for s in REGULARIZER_SETTINGS]
)


def _feasible(setting, weights):
    """Whether 'setting' is valid for K = weights.shape[1] (min_share needs K <= 1/m)."""
    name, kwargs = setting
    share = kwargs.get("min_share")
    if share is None:
        return True
    largest = max(
        (share.kwargs["start"], share.kwargs["end"])
        if isinstance(share, helpers.Scheduled)
        else (share,)
    )
    return largest * weights.shape[1] <= 1.0


@_SETTINGS
def test_penalty_and_gradient_match(setting, batch):
    """`penalty(x)` and its gradient w.r.t. x agree on every shared batch."""
    if not _feasible(setting, batch):
        pytest.skip("min_share is unsatisfiable for this K")
    name, kwargs = setting
    keras_reg = build("keras", name, kwargs)
    torch_reg = build("torch", name, kwargs)
    assert_penalties_equal(keras_reg, torch_reg, batch)


@pytest.mark.parametrize("epoch", [None, 0, 1, 2, 3, 4, 6, 20])
@pytest.mark.parametrize(
    "setting",
    [
        s
        for s in REGULARIZER_SETTINGS
        if any(isinstance(v, helpers.Scheduled) for v in s[1].values())
    ],
    ids=lambda s: setting_id(s),
)
def test_scheduled_penalty_matches_at_every_epoch(setting, epoch):
    """A scheduled hyperparameter reads the clock identically on both backends."""
    name, kwargs = setting
    keras_reg = build("keras", name, kwargs)
    torch_reg = build("torch", name, kwargs)
    keras_schedules.set_epoch(epoch)
    torch_schedules.set_epoch(epoch)
    assert_penalties_equal(keras_reg, torch_reg, BATCHES["random_k5"], f"epoch {epoch}")


def test_min_state_size_rejects_an_infeasible_floor_in_both():
    """The satisfiability check raises the same error at call time."""
    weights = np.full((10, 5), 0.2, np.float32)
    for backend in ("keras", "torch"):
        reg = build(backend, "MinStateSize", dict(l2=1.0, min_share=0.25))
        penalty = reg.penalty
        with pytest.raises(ValueError, match="not satisfiable"):
            penalty(
                tf.constant(weights) if backend == "keras" else torch.tensor(weights)
            )


# -- cross-batch EMA --------------------------------------------------------

_EMA_SETTINGS = [s for s in REGULARIZER_SETTINGS if s[1].get("ema_decay") is not None]


@pytest.mark.parametrize(
    "setting", _EMA_SETTINGS, ids=[setting_id(s) for s in _EMA_SETTINGS]
)
def test_ema_sequence_matches_batch_by_batch(setting):
    """Six batches give equal smoothed statistics, steps, penalties and gradients."""
    name, kwargs = setting
    keras_reg = build("keras", name, kwargs)
    torch_reg = build("torch", name, kwargs)
    for step in range(1, 7):
        weights = random_weights(32, 5, seed=100 + step)
        assert_penalties_equal(keras_reg, torch_reg, weights, f"step {step}")
        assert torch_reg.smoothing_steps == keras_reg.smoothing_steps == step
        keras_stats = keras_reg.smoothed_statistics()
        torch_stats = torch_reg.smoothed_statistics()
        assert keras_stats.keys() == torch_stats.keys()
        for key, value in keras_stats.items():
            np.testing.assert_allclose(
                torch_stats[key].numpy(), value.numpy(), rtol=RTOL, atol=ATOL
            )


@pytest.mark.parametrize(
    "setting", _EMA_SETTINGS, ids=[setting_id(s) for s in _EMA_SETTINGS]
)
def test_eval_mode_does_not_advance_the_average(setting, monkeypatch):
    """With the estimator not training, both backends read without updating."""
    name, kwargs = setting
    keras_reg = build("keras", name, kwargs)
    torch_reg = build("torch", name, kwargs)
    for step in range(3):
        assert_penalties_equal(
            keras_reg, torch_reg, random_weights(32, 5, seed=200 + step)
        )

    monkeypatch.setattr(keras_regularizers, "_in_training", lambda: False)
    torch_reg.eval()
    for step in range(3):
        assert_penalties_equal(
            keras_reg, torch_reg, random_weights(32, 5, seed=300 + step), "eval"
        )
        assert torch_reg.smoothing_steps == keras_reg.smoothing_steps == 3


def test_ema_before_any_training_batch_reads_the_batch(monkeypatch):
    """In eval mode from the start there is no history: the batch value is used."""
    kwargs = dict(l2=1.0, target=2.0, ema_decay=0.9)
    keras_reg = build("keras", "DegreesOfFreedom", kwargs)
    torch_reg = build("torch", "DegreesOfFreedom", kwargs)
    monkeypatch.setattr(keras_regularizers, "_in_training", lambda: False)
    torch_reg.eval()
    weights = random_weights(32, 5, seed=7)
    assert_penalties_equal(keras_reg, torch_reg, weights)
    plain = build("torch", "DegreesOfFreedom", dict(l2=1.0, target=2.0))
    np.testing.assert_allclose(
        helpers.torch_penalty(torch_reg, weights)[0],
        helpers.torch_penalty(plain, weights)[0],
        rtol=RTOL,
    )
