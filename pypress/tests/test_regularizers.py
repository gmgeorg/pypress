import numpy as np
import tensorflow as tf

# Import the regularizers from your module.
# Adjust the import path according to your package structure.
from pypress.keras.regularizers import (
    DegreesOfFreedom,
    Uniform,
    CombinedRegularizer,
    UniformAndDegreesOfFreedomRegularizer,
)

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


def test_combined_regularizer_tuples():
    # Create a dummy weight matrix.
    # For example, a 2 x 3 matrix where rows represent outputs/features.
    x = tf.constant([[0.1, 0.2, 0.7], [0.3, 0.3, 0.4]], dtype=tf.float32)

    # Instantiate individual regularizers directly.
    uniform_reg = Uniform(l1=0.01)
    df_reg = DegreesOfFreedom(l1=0.02, df=1.0)

    # Expected penalty: the sum of the individual penalties.
    expected_penalty = uniform_reg(x) + df_reg(x)

    # Create the combined regularizer using a list of tuples: (constructor, kwargs)
    regularizer_tuples = [
        (Uniform, {"l1": 0.01}),
        (DegreesOfFreedom, {"l1": 0.02, "df": 1.0}),
    ]
    combined_reg = CombinedRegularizer(regularizer_tuples=regularizer_tuples)

    # Compute the combined penalty.
    combined_penalty = combined_reg(x)

    # Assert that the combined penalty equals the sum of the individual penalties.
    np.testing.assert_allclose(
        combined_penalty.numpy(),
        expected_penalty.numpy(),
        atol=1e-6,
        err_msg="CombinedRegularizer (tuples) does not match sum of individual regularizers.",
    )


# def test_composite_regularizer_serialization():
#     # Create a dummy weight tensor.
#     # For example, a 2 x 3 matrix.
#     x = tf.constant([[0.1, 0.2, 0.7],
#                      [0.3, 0.3, 0.4]], dtype=tf.float32)

#     # Define a list of tuples: (constructor, kwargs)
#     regularizer_tuples = [
#         (Uniform, {"l1": 0.01}),
#         (DegreesOfFreedom, {"l1": 0.02, "df": 1.0})
#     ]

#     # Instantiate the CompositeRegularizer.
#     composite_reg_orig = CombinedRegularizer(regularizer_tuples=regularizer_tuples)

#     # Compute penalty from the original instance.
#     orig_penalty = composite_reg_orig(x)

#     # Serialize the composite regularizer (get config).
#     config = composite_reg_orig.get_config()

#     # Deserialize the composite regularizer.
#     # Because we registered CompositeRegularizer with tf.keras.utils.register_keras_serializable,
#     # we can use tf.keras.regularizers.deserialize().
#     composite_reg_new = tf.keras.regularizers.deserialize(config)

#     # Compute penalty from the deserialized instance.
#     new_penalty = composite_reg_new(x)

#     # Verify that both penalties are identical (within a small tolerance).
#     np.testing.assert_allclose(
#         new_penalty.numpy(), orig_penalty.numpy(), atol=1e-6,
#         err_msg="Deserialized CompositeRegularizer does not match original instance."
#     )


def test_combined_regularizer_penalty():
    """
    Test that the combined regularizer returns the sum of the Uniform and
    DegreesOfFreedom penalties.
    """
    # Create a dummy weight tensor (e.g., a 2x3 matrix)
    x = tf.constant([[0.1, 0.2, 0.7], [0.3, 0.3, 0.4]], dtype=tf.float32)

    # Instantiate our combined regularizer with explicit parameters.
    combined_reg = UniformAndDegreesOfFreedomRegularizer(
        uniform_l1=0.01, dof_l1=0.02, dof_target=1.0
    )

    # Also instantiate the two individual regularizers directly.
    uniform_reg = Uniform(l1=0.01)
    dof_reg = DegreesOfFreedom(l1=0.02, df=1.0)

    # Compute expected penalty as the sum of the two individual penalties.
    expected_penalty = uniform_reg(x) + dof_reg(x)
    computed_penalty = combined_reg(x)

    np.testing.assert_allclose(
        computed_penalty.numpy(),
        expected_penalty.numpy(),
        atol=1e-6,
        err_msg="Combined regularizer penalty does not equal the sum of the individual penalties.",
    )


def test_combined_regularizer_get_config():
    """
    Test that get_config returns the expected configuration dictionary.
    """
    uniform_l1 = 0.01
    dof_l1 = 0.02
    dof_target = 1.0
    reg = UniformAndDegreesOfFreedomRegularizer(
        uniform_l1=uniform_l1, dof_l1=dof_l1, dof_target=dof_target
    )
    config = reg.get_config()
    # Check that the config returns the same values.
    assert config["uniform_l1"] == uniform_l1
    assert config["dof_l1"] == dof_l1
    assert config["dof_target"] == dof_target


def test_combined_regularizer_serialization_deserialization():
    """
    Test that serializing and then deserializing the regularizer yields an object
    that produces the same penalty on a given tensor.
    """
    reg_orig = UniformAndDegreesOfFreedomRegularizer(
        uniform_l1=0.01, dof_l1=0.02, dof_target=1.0
    )
    config = reg_orig.get_config()
    # Deserialize using the class from_config method.
    reg_new = UniformAndDegreesOfFreedomRegularizer.from_config(config)

    # Create a dummy weight tensor.
    x = tf.constant([[0.1, 0.2, 0.7], [0.3, 0.3, 0.4]], dtype=tf.float32)

    # Both the original and deserialized regularizer should produce the same output.
    np.testing.assert_allclose(
        reg_new(x).numpy(),
        reg_orig(x).numpy(),
        atol=1e-6,
        err_msg="Deserialized regularizer does not produce the same penalty as the original.",
    )
