"""pypress.utils keeps working as a compatibility layer over the new modules."""

import numpy as np
import pytest
import tensorflow as tf

from pypress import analysis, utils
from pypress.keras import initializers, statistics


def test_moved_functions_are_re_exported():
    """Old import paths resolve to the functions in their new homes."""
    assert utils.state_size is analysis.state_size
    assert utils.col_normalize is analysis.col_normalize
    assert utils.agg_data_by_state is analysis.agg_data_by_state
    assert utils.kernel_matrix is analysis.smoother_matrix
    assert utils.initialize_from_y is initializers.initialize_from_y
    assert utils.safe_log is statistics.safe_log
    assert utils.safe_xlogx is statistics.safe_xlogx


def test_old_import_paths_still_work():
    """`from pypress.utils import ...` keeps working for the moved functions."""
    from pypress.utils import initialize_from_y, kernel_matrix  # noqa: F401


WEIGHTS = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])


@pytest.mark.parametrize(
    "name, expected",
    [
        ("tf_state_size", statistics.state_sizes(WEIGHTS)),
        ("tf_col_normalize", WEIGHTS / statistics.state_sizes(WEIGHTS)),
        (
            "tr_kernel",
            np.trace(analysis.smoother_matrix(WEIGHTS.numpy())),
        ),
    ],
)
def test_deprecated_graph_helpers_warn_and_still_work(name, expected):
    """The tests-only graph helpers warn and return what they used to."""
    with pytest.warns(DeprecationWarning, match=name):
        result = getattr(utils, name)(WEIGHTS)
    np.testing.assert_allclose(np.asarray(result), np.asarray(expected), rtol=1e-6)


def test_deprecated_tr_kernel_terms_warns_and_still_works():
    """tr_kernel_terms returns (numerator, state sizes)."""
    with pytest.warns(DeprecationWarning, match="tr_kernel_terms"):
        numerator, denominator = utils.tr_kernel_terms(WEIGHTS)
    np.testing.assert_allclose(numerator, statistics.trace_numerator(WEIGHTS))
    np.testing.assert_allclose(denominator, statistics.state_sizes(WEIGHTS))
