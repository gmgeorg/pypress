"""Shared batches and tolerances for the parity tests."""

import numpy as np

# Float32 rounding differs between the two backends by a few ulps.
RTOL = 2e-5
ATOL = 2e-6


def random_weights(n_rows, n_states, seed, concentration=0.7):
    """A float32 batch of predictive state weights (softmax of random logits)."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(scale=2.0 / concentration, size=(n_rows, n_states))
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    return (weights / weights.sum(axis=1, keepdims=True)).astype(np.float32)


def one_hot_weights(n_rows, n_states, seed):
    """Hard assignments."""
    rng = np.random.default_rng(seed)
    weights = np.zeros((n_rows, n_states), dtype=np.float32)
    weights[np.arange(n_rows), rng.integers(0, n_states, size=n_rows)] = 1.0
    return weights


def empty_state_weights(n_rows, n_states, seed):
    """Random weights with the last state exactly empty (a dead state)."""
    weights = random_weights(n_rows, n_states - 1, seed)
    return np.concatenate([weights, np.zeros((n_rows, 1), np.float32)], axis=1)


# name -> batch, covering random, one-hot, empty-state and K = 1 cases.
BATCHES = {
    "random_k5": random_weights(48, 5, seed=1),
    "random_k9": random_weights(33, 9, seed=2),
    "sharp_k4": random_weights(40, 4, seed=3, concentration=0.15),
    "one_hot_k6": one_hot_weights(30, 6, seed=4),
    "empty_state_k5": empty_state_weights(30, 5, seed=5),
    "single_state": np.ones((12, 1), np.float32),
    "uniform_k4": np.full((16, 4), 0.25, np.float32),
}
