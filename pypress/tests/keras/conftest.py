"""Fixtures for the Keras tests; the directory is skipped without TensorFlow."""

import pytest

pytest.importorskip("tensorflow")
