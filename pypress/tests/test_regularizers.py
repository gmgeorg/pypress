import numpy as np
import tensorflow as tf

# Import the regularizers from your module.
# Adjust the import path according to your package structure.
from pypress.keras.regularizers import (
    DegreesOfFreedom,
    TargetEntropy,
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
    l2 = 1.0
    n_states = 3  # number of columns
    units = 1  # number of rows
    # Construct a weight matrix with each row uniform: [1/3, 1/3, 1/3]
    weights = np.full((units, n_states), 1.0 / n_states, dtype=np.float32)
    # Instantiate the regularizer
    reg = Uniform(l2=l2)
    penalty = reg(tf.convert_to_tensor(weights))

    # Expected penalty: l2 * (log(3) - log(3)) ** 2 = 0
    np.testing.assert_allclose(penalty.numpy(), 0.0, atol=1e-3)


def test_uniform_regularizer_nonuniform():
    """
    Test the Uniform regularizer with a non-uniform row.
    For example, a row [0.8, 0.1, 0.1] has an entropy lower than log(3),
    so the penalty (squared deviation from log(3)) will be positive.
    """
    l2 = 1.0
    n_states = 3
    # Create two rows: one uniform and one non-uniform.
    row_uniform = np.full((1, n_states), 1.0 / n_states, dtype=np.float32)
    row_nonuniform = np.array([[0.8, 0.1, 0.1]], dtype=np.float32)
    weights = np.concatenate([row_uniform, row_nonuniform], axis=0)

    reg = Uniform(l2=l2)
    penalty = reg(tf.convert_to_tensor(weights))

    # Since the uniform row contributes zero penalty, the overall mean entropy is < log(3)
    # and penalty should be positive.
    assert penalty.numpy() > 0.0

    # Uniform uses a squared (L2) penalty: l2 * (log(3) - mean_entropy) ** 2.
    row_nonuniform_entropy = -np.sum(row_nonuniform * np.log(row_nonuniform))
    mean_entropy = (np.log(n_states) + row_nonuniform_entropy) / 2
    expected_penalty = l2 * (np.log(n_states) - mean_entropy) ** 2
    np.testing.assert_allclose(penalty.numpy(), expected_penalty, atol=1e-5)


def test_uniform_is_special_case_of_target_entropy():
    """Uniform(l2) should equal TargetEntropy(l2, target=log(K)) for K = x.shape[1]."""
    n_states = 4
    weights = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    x = tf.convert_to_tensor(weights)

    assert isinstance(Uniform(), TargetEntropy)

    uniform_penalty = Uniform(l2=1.0)(x)
    target_entropy_penalty = TargetEntropy(l2=1.0, target=np.log(n_states))(x)
    np.testing.assert_allclose(
        uniform_penalty.numpy(), target_entropy_penalty.numpy(), atol=1e-6
    )


# -------- Tests for the TargetEntropy Regularizer --------


def test_target_entropy_default_target_is_log_half_k():
    """With no explicit target, TargetEntropy targets log(0.5 * K)."""
    n_states = 4
    # Uniform row: entropy == log(K).
    weights = np.full((1, n_states), 1.0 / n_states, dtype=np.float32)
    reg = TargetEntropy(l2=1.0)
    penalty = reg(tf.convert_to_tensor(weights))

    target = np.log(0.5 * n_states)
    expected_penalty = (np.log(n_states) - target) ** 2
    np.testing.assert_allclose(penalty.numpy(), expected_penalty, atol=1e-5)


def test_target_entropy_explicit_target_zero_penalty():
    """Penalty is zero when the row's entropy already matches the explicit target."""
    n_states = 3
    weights = np.full((1, n_states), 1.0 / n_states, dtype=np.float32)
    reg = TargetEntropy(l2=1.0, target=float(np.log(n_states)))
    penalty = reg(tf.convert_to_tensor(weights))
    np.testing.assert_allclose(penalty.numpy(), 0.0, atol=1e-5)


def test_target_entropy_clamps_active_states_at_one():
    """For K=1 (single, deterministic state), the dynamic target is clamped to log(1) = 0."""
    weights = np.array([[1.0]], dtype=np.float32)
    reg = TargetEntropy(l2=1.0)
    penalty = reg(tf.convert_to_tensor(weights))
    # Entropy of a single deterministic state is 0, and the clamped target is also 0.
    np.testing.assert_allclose(penalty.numpy(), 0.0, atol=1e-6)


def test_target_entropy_get_config():
    """get_config omits 'target' when unset, includes it when set explicitly."""
    assert TargetEntropy(l2=0.5).get_config() == {"l2": 0.5}
    assert TargetEntropy(l2=0.5, target=0.3).get_config() == {"l2": 0.5, "target": 0.3}


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
    uniform_reg = Uniform(l2=0.01)
    df_reg = DegreesOfFreedom(l1=0.02, df=1.0)

    # Expected penalty: the sum of the individual penalties.
    expected_penalty = uniform_reg(x) + df_reg(x)

    # Create the combined regularizer using a list of tuples: (constructor, kwargs)
    regularizer_tuples = [
        (Uniform, {"l2": 0.01}),
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
        uniform_l2=0.01, dof_l1=0.02, dof_target=1.0
    )

    # Also instantiate the two individual regularizers directly.
    uniform_reg = Uniform(l2=0.01)
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
    uniform_l2 = 0.01
    dof_l1 = 0.02
    dof_target = 1.0
    reg = UniformAndDegreesOfFreedomRegularizer(
        uniform_l2=uniform_l2, dof_l1=dof_l1, dof_target=dof_target
    )
    config = reg.get_config()
    # Check that the config returns the same values.
    assert config["uniform_l2"] == uniform_l2
    assert config["dof_l1"] == dof_l1
    assert config["dof_target"] == dof_target


def test_combined_regularizer_serialization_deserialization():
    """
    Test that serializing and then deserializing the regularizer yields an object
    that produces the same penalty on a given tensor.
    """
    reg_orig = UniformAndDegreesOfFreedomRegularizer(
        uniform_l2=0.01, dof_l1=0.02, dof_target=1.0
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
