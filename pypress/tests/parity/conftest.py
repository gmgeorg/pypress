"""Fixtures for the parity tests; the directory is skipped without both backends."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("tensorflow")

from .data import BATCHES  # noqa: E402


@pytest.fixture(params=sorted(BATCHES))
def batch(request):
    """Each shared batch of predictive state weights, as a float32 numpy array."""
    return BATCHES[request.param]
