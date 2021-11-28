"""Module for testing utils module."""

from typing import Tuple

import numpy as np
import pandas as pd

from ..keras import utils


def _test_weights() -> Tuple[pd.DataFrame, pd.Series]:
    weights = pd.DataFrame(
        {"0": [0.1, 0.2, 0.8], "1": [0.3, 0.0, 0.1], "2": [0.6, 0.8, 0.1]},
        index=["foo", "bar", "hello"],
    )

    expected_size = pd.Series([1.1, 0.4, 1.5], index=["0", "1", "2"])
    expected_size.index.name = "state"
    return weights, expected_size


def test_state_size():
    w, expected_size = _test_weights()
    pd.testing.assert_series_equal(utils.size_state(w), expected_size)


def test_col_normalizer():
    w, expected_size = _test_weights()
    norm_w = utils.col_normalize(w)

    col_sums_expected = pd.Series([1.0, 1.0, 1.0], index=["0", "1", "2"])
    col_sums_expected.index.name = "state"
    pd.testing.assert_series_equal(norm_w.sum(axis=0), col_sums_expected)


def test_tf_state_size():
    w, _ = _test_weights()
    tf_s = utils.tf_size_state(w)
    s = utils.size_state(w)
    np.testing.assert_allclose(s.values, tf_s.numpy())


def test_tf_col_normalize():
    w, _ = _test_weights()
    tf_nw = utils.tf_col_normalize(w)
    nw = utils.col_normalize(w)
    np.testing.assert_allclose(nw.values, tf_nw.numpy())
