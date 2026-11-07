"""Fixtures for the torch-only tests; the directory is skipped without PyTorch."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def simplex_weights(n_rows, n_states, seed=0, concentration=0.7):
    """A float32 batch of predictive state weights (rows on the simplex)."""
    rng = np.random.default_rng(seed)
    return torch.tensor(
        rng.dirichlet(np.full(n_states, concentration), size=n_rows),
        dtype=torch.float32,
    )


@pytest.fixture
def weights():
    """A random batch of predictive state weights, 64 rows by 5 states."""
    return simplex_weights(64, 5)
