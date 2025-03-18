import numpy as np
import tensorflow as tf

# Import the regularizers from your module.
# Adjust the import path according to your package structure.
from pypress.keras.regularizers import DegreesOfFreedom, Uniform

_EPS = 1e-6

# -------- Tests for the Uniform Regularizer --------


def test_uniform_regularizer_uniform():
    """
    Test the Uniform regularizer with a weight matrix whose rows are exactly uniform.
    For a row of length J where every element is 1/J, the Shannon entropy is log(J).
    Thus the penalty should be zero.
    """
    l1 = 1.0
    n_states = 3  # number of columns
    units = 1  # number of rows
    # Construct a weight matrix with each row uniform: [1/3, 1/3, 1/3]
    weights = np.full((units, n_states), 1.0 / n_states, dtype=np.float32)
    # Instantiate the regularizer
    reg = Uniform(l1=l1)
    penalty = reg(tf.convert_to_tensor(weights))

    # Expected penalty: l1 * (log(3) - log(3)) = 0
    np.testing.assert_allclose(penalty.numpy(), 0.0, atol=1e-3)


def test_uniform_regularizer_nonuniform():
    """
    Test the Uniform regularizer with a non-uniform row.
    For example, a row [0.8, 0.1, 0.1] has an entropy lower than log(3),
    so the penalty (log(3) - entropy) will be positive.
    """
    l1 = 1.0
    n_states = 3
    # Create two rows: one uniform and one non-uniform.
    row_uniform = np.full((1, n_states), 1.0 / n_states, dtype=np.float32)
    row_nonuniform = np.array([[0.8, 0.1, 0.1]], dtype=np.float32)
    weights = np.concatenate([row_uniform, row_nonuniform], axis=0)

    reg = Uniform(l1=l1)
    penalty = reg(tf.convert_to_tensor(weights))

    # Since the uniform row contributes zero penalty, the overall mean entropy is < log(3)
    # and penalty should be positive.
    assert penalty.numpy() > 0.0


# -------- Tests for the DegreesOfFreedom Regularizer --------


def test_degrees_of_freedom_regularizer_zero_penalty():
    """
    Test DegreesOfFreedom regularizer when the target degrees of freedom equals the trace of the kernel.
    For a weight matrix where each column already has unit norm, tr_kernel should equal the number of states.
    """
    l1 = 1.0
    df_target = 2.0
    # Create a 2x2 identity matrix.
    weights = np.eye(2, dtype=np.float32)
    # Assuming tf_col_normalize normalizes columns by L2 norm, the columns of identity remain the same.
    # Then, tr_kernel(weights) should equal 1 + 1 = 2.

    reg = DegreesOfFreedom(l1=l1, df=df_target)
    penalty = reg(tf.convert_to_tensor(weights))

    # Expected penalty: l1 * abs(2 - 2) = 0.
    np.testing.assert_allclose(penalty.numpy(), 0.0, atol=1e-6)


def test_degrees_of_freedom_regularizer_nonzero_penalty():
    """
    Test DegreesOfFreedom regularizer for a weight matrix where the sum of L2 norms of columns
    deviates from the target degrees of freedom.

    For example, consider a 2x2 matrix:
        [[1, 2],
         [0, 0]]
    The L2 norm of the first column is 1, and for the second column is 2.
    Thus, tr_kernel(weights) is expected to be 1 + 2 = 3.
    With a target df of 2, the penalty should be l1 * |3 - 2| = l1 * 1.
    """
    l1 = 1.0
    df_target = 2.0
    weights = np.array([[1, 2], [0, 0]], dtype=np.float32)

    reg = DegreesOfFreedom(l1=l1, df=df_target)
    penalty = reg(tf.convert_to_tensor(weights))

    # Expected penalty is 1.0 * |3 - 2| = 1.0
    np.testing.assert_allclose(penalty.numpy(), 1.0, atol=1e-6)
