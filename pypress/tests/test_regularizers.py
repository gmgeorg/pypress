import numpy as np
import pytest
import tensorflow as tf

# Import the regularizers from your module.
# Adjust the import path according to your package structure.
from pypress.keras.schedules import ScheduledValue
from pypress.keras.regularizers import (
    DegreesOfFreedom,
    MinStateSize,
    StateSizeEntropy,
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
    so the penalty (squared, K-normalized deviation from log(3)) will be positive.
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

    # Uniform uses a squared, K-normalized (L2) penalty:
    # l2 * ((log(3) - mean_entropy) / log(3)) ** 2.
    row_nonuniform_entropy = -np.sum(row_nonuniform * np.log(row_nonuniform))
    mean_entropy = (np.log(n_states) + row_nonuniform_entropy) / 2
    expected_penalty = l2 * ((np.log(n_states) - mean_entropy) / np.log(n_states)) ** 2
    np.testing.assert_allclose(penalty.numpy(), expected_penalty, atol=1e-5)


def test_uniform_regularizer_scale_invariant_with_k():
    """The Uniform penalty for a fully deterministic row is bounded by l2,
    regardless of K -- it should not grow as log(K)^2."""
    l2 = 1.0
    for n_states in (3, 10, 100):
        row = np.zeros((1, n_states), dtype=np.float32)
        row[0, 0] = 1.0  # fully deterministic row: entropy = 0
        penalty = Uniform(l2=l2)(tf.convert_to_tensor(row))
        # Normalized deviation is exactly 1.0 at zero entropy, so penalty == l2.
        np.testing.assert_allclose(penalty.numpy(), l2, atol=1e-5)


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
    expected_penalty = ((np.log(n_states) - target) / np.log(n_states)) ** 2
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
    l2 = 1.0
    df_target = 2.0
    # Create a 2x2 identity matrix.
    weights = np.eye(2, dtype=np.float32)
    # Assuming tf_col_normalize normalizes columns by L2 norm, the columns of identity remain the same.
    # Then, tr_kernel(weights) should equal 1 + 1 = 2.

    reg = DegreesOfFreedom(l2=l2, df=df_target)
    penalty = reg(tf.convert_to_tensor(weights))

    # Expected penalty: l2 * (2 - 2) ** 2 = 0.
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
    With a target df of 2, the penalty should be l2 * (3 - 2) ** 2 = l2 * 1.
    """
    l2 = 1.0
    df_target = 2.0
    weights = np.array([[1, 2], [0, 0]], dtype=np.float32)

    reg = DegreesOfFreedom(l2=l2, df=df_target)
    penalty = reg(tf.convert_to_tensor(weights))

    # Expected penalty is 1.0 * (3 - 2) ** 2 = 1.0
    np.testing.assert_allclose(penalty.numpy(), 1.0, atol=1e-6)


def test_normalized_degrees_of_freedom_is_scale_invariant_with_k():
    """Equal fractional trace errors cost equally with absolute DoF targets."""
    penalties = []
    for n_states, n_alive, target in ((10, 3, 5.0), (20, 6, 10.0)):
        x = _hard_assignment_weights(n_states=n_states, n_alive=n_alive)
        penalties.append(
            DegreesOfFreedom(l2=1.0, target=target, normalize=True)(x).numpy()
        )

    # (3 - 5) / 10 == (6 - 10) / 20 == -0.2.
    np.testing.assert_allclose(penalties, [0.04, 0.04], atol=1e-5)


def test_normalized_degrees_of_freedom_serializes():
    """The normalization choice and an absolute scheduled target survive config."""
    target = ScheduledValue(start=9.0, end=4.0, duration_epochs=3)
    reg = DegreesOfFreedom(l2=0.2, target=target, normalize=True)
    restored = DegreesOfFreedom.from_config(reg.get_config())
    assert restored.normalize is True
    assert isinstance(restored.target, ScheduledValue)
    assert restored.target.get_config() == target.get_config()


def test_combined_regularizer_tuples():
    # Create a dummy weight matrix.
    # For example, a 2 x 3 matrix where rows represent outputs/features.
    x = tf.constant([[0.1, 0.2, 0.7], [0.3, 0.3, 0.4]], dtype=tf.float32)

    # Instantiate individual regularizers directly.
    uniform_reg = Uniform(l2=0.01)
    df_reg = DegreesOfFreedom(l2=0.02, df=1.0)

    # Expected penalty: the sum of the individual penalties.
    expected_penalty = uniform_reg(x) + df_reg(x)

    # Create the combined regularizer using a list of tuples: (constructor, kwargs)
    regularizer_tuples = [
        (Uniform, {"l2": 0.01}),
        (DegreesOfFreedom, {"l2": 0.02, "df": 1.0}),
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
#         (Uniform, {"l2": 0.01}),
#         (DegreesOfFreedom, {"l2": 0.02, "df": 1.0})
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
        uniform_l2=0.01, dof_l2=0.02, dof_target=1.0
    )

    # Also instantiate the two individual regularizers directly.
    uniform_reg = Uniform(l2=0.01)
    dof_reg = DegreesOfFreedom(l2=0.02, df=1.0)

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
    dof_l2 = 0.02
    dof_target = 1.0
    reg = UniformAndDegreesOfFreedomRegularizer(
        uniform_l2=uniform_l2, dof_l2=dof_l2, dof_target=dof_target
    )
    config = reg.get_config()
    # Check that the config returns the same values.
    assert config["uniform_l2"] == uniform_l2
    assert config["dof_l2"] == dof_l2
    assert config["dof_target"] == dof_target


def test_combined_regularizer_serialization_deserialization():
    """
    Test that serializing and then deserializing the regularizer yields an object
    that produces the same penalty on a given tensor.
    """
    reg_orig = UniformAndDegreesOfFreedomRegularizer(
        uniform_l2=0.01, dof_l2=0.02, dof_target=1.0
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


# -------- Tests for the StateSizeEntropy Regularizer --------


def _hard_assignment_weights(n_states: int, n_alive: int, n_rows: int = 200, eps=1e-6):
    """Weight matrix where rows are near-deterministic over only 'n_alive' states."""
    rng = np.random.default_rng(0)
    weights = np.full((n_rows, n_states), eps, dtype=np.float32)
    for row, col in enumerate(rng.integers(0, n_alive, n_rows)):
        weights[row, col] = 1.0
    weights /= weights.sum(axis=1, keepdims=True)
    return tf.convert_to_tensor(weights)


def test_state_size_entropy_zero_for_uniform_usage():
    """Equally used states have a marginal entropy of log(K): no penalty."""
    n_states = 6
    # Deterministic rows, but each state used by exactly the same number of rows.
    weights = np.zeros((n_states * 5, n_states), dtype=np.float32)
    for row in range(weights.shape[0]):
        weights[row, row % n_states] = 1.0

    penalty = StateSizeEntropy(l2=1.0)(tf.convert_to_tensor(weights))
    np.testing.assert_allclose(penalty.numpy(), 0.0, atol=1e-6)


def test_state_size_entropy_penalizes_dead_states():
    """Dead states cost, and cost more the more of them there are."""
    reg = StateSizeEntropy(l2=1.0)
    honest = reg(_hard_assignment_weights(n_states=5, n_alive=5)).numpy()
    collapsed = reg(_hard_assignment_weights(n_states=50, n_alive=5)).numpy()

    assert collapsed > honest
    # 45 of 50 states dead should be an order of magnitude worse than 0 of 5.
    assert collapsed > 10 * honest


def test_state_size_entropy_sees_what_dof_and_row_entropy_miss():
    """The failure mode both existing regularizers are blind to.

    K=5 fully used and K=50 with 45 dead states have the same kernel trace and
    (essentially) the same mean row entropy, so neither DegreesOfFreedom nor
    TargetEntropy can tell them apart. StateSizeEntropy must.
    """
    honest = _hard_assignment_weights(n_states=5, n_alive=5)
    collapsed = _hard_assignment_weights(n_states=50, n_alive=5)

    # Both existing penalties are (near) blind: same df, same mean row entropy.
    dof = DegreesOfFreedom(l2=1.0, target=1.0)
    np.testing.assert_allclose(dof(honest).numpy(), dof(collapsed).numpy(), rtol=1e-3)
    row_entropy = TargetEntropy(l2=1.0, target=0.0)
    np.testing.assert_allclose(
        row_entropy(honest).numpy(), row_entropy(collapsed).numpy(), atol=1e-3
    )

    # The new penalty separates them clearly.
    reg = StateSizeEntropy(l2=1.0)
    assert reg(collapsed).numpy() - reg(honest).numpy() > 0.4


def test_state_size_entropy_bounded_by_l2():
    """Penalty is bounded in [0, l2] for any K -- full collapse hits exactly l2."""
    l2 = 0.5
    for n_states in (3, 10, 100):
        # All population mass on a single state: marginal entropy is 0.
        weights = np.zeros((4, n_states), dtype=np.float32)
        weights[:, 0] = 1.0
        penalty = StateSizeEntropy(l2=l2)(tf.convert_to_tensor(weights))
        np.testing.assert_allclose(penalty.numpy(), l2, atol=1e-5)


def test_state_size_entropy_active_fraction_is_one_sided():
    """With active_fraction < 1, usage above the floor is not penalized."""
    # 5 of 10 states used uniformly: exp(H(p)) == 5 == 0.5 * K, exactly at target.
    weights = np.zeros((20, 10), dtype=np.float32)
    for row in range(weights.shape[0]):
        weights[row, row % 5] = 1.0
    weights = tf.convert_to_tensor(weights)

    np.testing.assert_allclose(
        StateSizeEntropy(l2=1.0, active_fraction=0.5)(weights).numpy(), 0.0, atol=1e-6
    )
    # Using *more* states than the floor stays unpenalized ...
    all_used = np.zeros((20, 10), dtype=np.float32)
    for row in range(all_used.shape[0]):
        all_used[row, row % 10] = 1.0
    np.testing.assert_allclose(
        StateSizeEntropy(l2=1.0, active_fraction=0.5)(
            tf.convert_to_tensor(all_used)
        ).numpy(),
        0.0,
        atol=1e-6,
    )
    # ... while active_fraction=1.0 still penalizes the half-used matrix.
    assert StateSizeEntropy(l2=1.0)(weights).numpy() > 0.0


def test_state_size_entropy_equals_normalized_kl_from_uniform():
    """For active_fraction=1 the penalty is l2 * KL(p || Uniform_K) / log(K)."""
    weights = np.array(
        [[0.7, 0.2, 0.1], [0.6, 0.3, 0.1], [0.8, 0.15, 0.05]], dtype=np.float32
    )
    l2 = 0.3
    penalty = StateSizeEntropy(l2=l2)(tf.convert_to_tensor(weights)).numpy()

    shares = weights.sum(axis=0) / weights.sum()
    kl = np.sum(shares * np.log(shares * weights.shape[1]))
    np.testing.assert_allclose(penalty, l2 * kl / np.log(weights.shape[1]), atol=1e-6)


def test_state_size_entropy_gradient_revives_dead_state():
    """The gradient must push mass *into* an underused state."""
    weights = tf.Variable(
        np.array([[0.98, 0.01, 0.01], [0.98, 0.01, 0.01]], dtype=np.float32)
    )
    with tf.GradientTape() as tape:
        penalty = StateSizeEntropy(l2=1.0)(weights)
    grad = tape.gradient(penalty, weights).numpy()

    # Increasing the starved columns lowers the penalty (negative gradient),
    # increasing the dominant one raises it.
    assert np.all(grad[:, 1:] < 0.0)
    assert np.all(grad[:, 0] > 0.0)
    assert np.all(np.isfinite(grad))


def test_state_size_entropy_handles_single_state_and_zero_columns():
    """K == 1 and exactly-zero columns must not produce NaN/inf."""
    single = StateSizeEntropy(l2=1.0)(tf.constant([[1.0], [1.0]], dtype=tf.float32))
    np.testing.assert_allclose(single.numpy(), 0.0, atol=1e-6)

    # Exact float zeros in a column (softmax underflow) stay finite.
    x = tf.Variable(np.array([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]], dtype=np.float32))
    with tf.GradientTape() as tape:
        penalty = StateSizeEntropy(l2=1.0)(x)
    assert np.isfinite(penalty.numpy())
    assert np.all(np.isfinite(tape.gradient(penalty, x).numpy()))


def test_state_size_entropy_invalid_active_fraction():
    """active_fraction outside (0, 1] is rejected."""
    for bad in (0.0, -0.5, 1.5):
        with pytest.raises(ValueError, match="active_fraction"):
            StateSizeEntropy(l2=1.0, active_fraction=bad)


def test_state_size_entropy_get_config_roundtrip():
    """get_config/from_config round-trips and reproduces the penalty."""
    reg = StateSizeEntropy(l2=0.25, active_fraction=0.75)
    config = reg.get_config()
    assert config == {"l2": 0.25, "active_fraction": 0.75}

    x = _hard_assignment_weights(n_states=8, n_alive=3)
    np.testing.assert_allclose(
        StateSizeEntropy.from_config(config)(x).numpy(), reg(x).numpy(), atol=1e-6
    )


def test_state_size_entropy_in_combined_regularizer():
    """StateSizeEntropy composes with CombinedRegularizer and survives from_config."""
    tuples = [
        (Uniform, {"l2": 0.01}),
        (DegreesOfFreedom, {"l2": 0.02, "target": 3.0}),
        (StateSizeEntropy, {"l2": 0.05, "active_fraction": 0.5}),
    ]
    combined = CombinedRegularizer(regularizer_tuples=tuples)
    x = _hard_assignment_weights(n_states=9, n_alive=3)

    expected = (
        Uniform(l2=0.01)(x)
        + DegreesOfFreedom(l2=0.02, target=3.0)(x)
        + StateSizeEntropy(l2=0.05, active_fraction=0.5)(x)
    )
    np.testing.assert_allclose(combined(x).numpy(), expected.numpy(), atol=1e-6)

    restored = CombinedRegularizer.from_config(combined.get_config())
    np.testing.assert_allclose(restored(x).numpy(), combined(x).numpy(), atol=1e-6)


# -------- Tests for the MinStateSize Regularizer --------


def _marginal_weights(shares):
    """A weight matrix whose column marginal is exactly 'shares'."""
    return tf.constant(np.array([shares], dtype=np.float32))


def test_min_state_size_zero_when_floor_is_met():
    """No penalty when every state clears the floor, however unequal they are."""
    reg = MinStateSize(l2=1.0, min_share=0.05)
    # Very unequal, but nothing below 5%: MinStateSize must not care.
    np.testing.assert_allclose(
        reg(_marginal_weights([0.60, 0.20, 0.10, 0.05, 0.05])).numpy(), 0.0, atol=1e-6
    )
    # Exactly at the floor is compliant.
    np.testing.assert_allclose(
        reg(_marginal_weights([0.05] * 20)).numpy(), 0.0, atol=1e-6
    )


def test_min_state_size_penalizes_below_floor():
    """Penalty appears below the floor and grows as the state shrinks."""
    reg = MinStateSize(l2=1.0, min_share=0.05)
    mild = reg(_marginal_weights([0.66, 0.30, 0.04])).numpy()
    severe = reg(_marginal_weights([0.699, 0.30, 0.001])).numpy()
    assert 0.0 < mild < severe


def test_min_state_size_matches_closed_form():
    """penalty == l2 * mean_k(max(0, 1 - p_k / min_share) ** 2)."""
    shares = np.array([0.70, 0.25, 0.03, 0.02], dtype=np.float64)
    l2, min_share = 0.4, 0.05
    penalty = MinStateSize(l2=l2, min_share=min_share)(
        _marginal_weights(shares.astype(np.float32))
    ).numpy()
    expected = l2 * np.mean(np.maximum(0.0, 1.0 - shares / min_share) ** 2)
    np.testing.assert_allclose(penalty, expected, atol=1e-6)


def test_min_state_size_catches_what_state_size_entropy_misses():
    """The concrete case an entropy penalty cannot express.

    At K=9, one state at 0.1% with the rest sharing the remainder has *higher*
    marginal entropy than the worst configuration respecting a 5% floor, so
    StateSizeEntropy tuned for that floor scores it zero. MinStateSize must not.
    """
    n_states = 9
    shares = np.full(n_states, (1.0 - 0.001) / (n_states - 1), dtype=np.float32)
    shares[0] = 0.001
    x = _marginal_weights(shares)

    # active_fraction=0.5 is the setting matching a 5% floor at K=9.
    np.testing.assert_allclose(
        StateSizeEntropy(l2=1.0, active_fraction=0.5)(x).numpy(), 0.0, atol=1e-6
    )
    assert MinStateSize(l2=1.0, min_share=0.05)(x).numpy() > 0.0


def test_min_state_size_bounded_and_scale_invariant_in_k():
    """Full collapse approaches l2 * (K - 1) / K, independent of K otherwise."""
    l2 = 0.5
    for n_states in (3, 10, 100):
        shares = np.zeros(n_states, dtype=np.float32)
        shares[0] = 1.0
        penalty = MinStateSize(l2=l2, min_share=1.0 / n_states)(
            _marginal_weights(shares)
        ).numpy()
        expected = l2 * (n_states - 1) / n_states
        np.testing.assert_allclose(penalty, expected, atol=1e-5)
        assert penalty < l2


def test_min_state_size_default_is_half_fair_share():
    """min_share=None defaults to 0.5 / K, computed dynamically from K."""
    n_states = 8
    # Exactly half the fair share in one state -> at the default floor, no penalty.
    shares = np.full(n_states, 0.0, dtype=np.float32)
    shares[0] = 0.5 / n_states
    shares[1:] = (1.0 - shares[0]) / (n_states - 1)
    x = _marginal_weights(shares)

    np.testing.assert_allclose(MinStateSize(l2=1.0)(x).numpy(), 0.0, atol=1e-6)
    np.testing.assert_allclose(
        MinStateSize(l2=1.0)(x).numpy(),
        MinStateSize(l2=1.0, min_share=0.5 / n_states)(x).numpy(),
        atol=1e-6,
    )


def test_min_state_size_rejects_infeasible_floor():
    """A floor that cannot be met by K states fails loudly, with a usable message."""
    reg = MinStateSize(l2=1.0, min_share=0.25)  # needs K <= 4
    with pytest.raises(ValueError, match="not satisfiable"):
        reg(_marginal_weights([0.2] * 5))
    # Feasible at K = 4.
    np.testing.assert_allclose(
        reg(_marginal_weights([0.25] * 4)).numpy(), 0.0, atol=1e-6
    )


def test_min_state_size_rejects_invalid_min_share():
    """min_share outside (0, 1] is rejected at construction."""
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="min_share"):
            MinStateSize(l2=1.0, min_share=bad)


def test_min_state_size_gradient_revives_starved_state():
    """Gradient pushes mass into the starved state and is finite."""
    x = tf.Variable(
        np.array([[0.97, 0.02, 0.01], [0.97, 0.02, 0.01]], dtype=np.float32)
    )
    with tf.GradientTape() as tape:
        penalty = MinStateSize(l2=1.0, min_share=0.2)(x)
    grad = tape.gradient(penalty, x).numpy()

    assert np.all(grad[:, 1:] < 0.0)
    assert np.all(grad[:, 0] > 0.0)
    assert np.all(np.isfinite(grad))


def test_min_state_size_handles_zero_columns_and_single_state():
    """Exactly-zero columns and K == 1 stay finite in value and gradient."""
    single = MinStateSize(l2=1.0)(tf.constant([[1.0], [1.0]], dtype=tf.float32))
    np.testing.assert_allclose(single.numpy(), 0.0, atol=1e-6)

    x = tf.Variable(np.array([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]], dtype=np.float32))
    with tf.GradientTape() as tape:
        penalty = MinStateSize(l2=1.0, min_share=0.1)(x)
    assert np.isfinite(penalty.numpy())
    assert np.all(np.isfinite(tape.gradient(penalty, x).numpy()))


def test_min_state_size_get_config_roundtrip():
    """get_config/from_config round-trips, including the None default."""
    reg = MinStateSize(l2=0.25, min_share=0.05)
    assert reg.get_config() == {"l2": 0.25, "min_share": 0.05}
    x = _hard_assignment_weights(n_states=8, n_alive=3)
    np.testing.assert_allclose(
        MinStateSize.from_config(reg.get_config())(x).numpy(), reg(x).numpy(), atol=1e-6
    )

    default_reg = MinStateSize(l2=0.25)
    assert default_reg.get_config() == {"l2": 0.25, "min_share": None}
    np.testing.assert_allclose(
        MinStateSize.from_config(default_reg.get_config())(x).numpy(),
        default_reg(x).numpy(),
        atol=1e-6,
    )


def test_min_state_size_in_combined_regularizer():
    """MinStateSize composes with CombinedRegularizer and survives from_config."""
    tuples = [
        (Uniform, {"l2": 0.01}),
        (MinStateSize, {"l2": 0.05, "min_share": 0.05}),
    ]
    combined = CombinedRegularizer(regularizer_tuples=tuples)
    x = _hard_assignment_weights(n_states=9, n_alive=3)

    expected = Uniform(l2=0.01)(x) + MinStateSize(l2=0.05, min_share=0.05)(x)
    np.testing.assert_allclose(combined(x).numpy(), expected.numpy(), atol=1e-6)

    restored = CombinedRegularizer.from_config(combined.get_config())
    np.testing.assert_allclose(restored(x).numpy(), combined(x).numpy(), atol=1e-6)


# -------- Tests for EMA smoothing of the state-size marginal --------


def _compliant_batches(n_batches, batch_size, n_states=9, floor=0.05, seed=0):
    """Batches drawn from a population that exactly meets 'floor' in every state."""
    rng = np.random.default_rng(seed)
    population = np.full(n_states, (1.0 - floor) / (n_states - 1))
    population[0] = floor
    for _ in range(n_batches):
        idx = rng.choice(n_states, size=batch_size, p=population)
        batch = np.zeros((batch_size, n_states), dtype=np.float32)
        batch[np.arange(batch_size), idx] = 1.0
        yield tf.constant(batch)


def test_ema_defaults_to_off():
    """Without ema_decay the penalty is the plain per-batch estimate."""
    x = _hard_assignment_weights(n_states=9, n_alive=4)
    plain = MinStateSize(l2=1.0, min_share=0.05)
    assert plain.ema_decay is None
    assert "ema_decay" not in plain.get_config()
    np.testing.assert_allclose(
        plain(x).numpy(), MinStateSize(l2=1.0, min_share=0.05)(x).numpy(), atol=1e-7
    )


def test_ema_first_call_equals_the_batch_estimate():
    """Bias correction makes step 1 exactly the batch value, not a zero-dragged one."""
    x = _hard_assignment_weights(n_states=9, n_alive=4)
    smoothed = MinStateSize(l2=1.0, min_share=0.05, ema_decay=0.99)(x).numpy()
    plain = MinStateSize(l2=1.0, min_share=0.05)(x).numpy()
    np.testing.assert_allclose(smoothed, plain, rtol=1e-5)


def test_ema_tracks_the_bias_corrected_average():
    """The smoothed marginal matches an explicit bias-corrected EMA."""
    decay = 0.8
    reg = MinStateSize(l2=1.0, min_share=0.05, ema_decay=decay)
    ema = np.zeros(9, dtype=np.float64)
    for step, batch in enumerate(_compliant_batches(6, 64), start=1):
        reg(batch)
        ema = decay * ema + (1.0 - decay) * batch.numpy().sum(axis=0)
        smoother = reg._smoothers["state_sizes"]
        np.testing.assert_allclose(
            smoother.average.numpy(), ema, atol=1e-4, err_msg=f"step {step}"
        )
        np.testing.assert_allclose(float(smoother.step), step)


def test_ema_removes_most_of_the_minibatch_bias():
    """The point of the EMA: a compliant population should stop being penalized.

    The penalties are convex, so each minibatch estimate is biased upward and
    averaging over more batches does not remove it. Smoothing the marginal across
    batches does.
    """
    plain = MinStateSize(l2=1.0, min_share=0.05)
    smoothed = MinStateSize(l2=1.0, min_share=0.05, ema_decay=0.95)

    plain_penalties, smoothed_penalties = [], []
    for batch in _compliant_batches(300, 32):
        plain_penalties.append(float(plain(batch)))
        smoothed_penalties.append(float(smoothed(batch)))

    # Ignore the EMA's burn-in; compare once it has accumulated history.
    plain_mean = np.mean(plain_penalties[100:])
    smoothed_mean = np.mean(smoothed_penalties[100:])

    # True penalty is 0: the batch estimate is badly biased, the smoothed one is not.
    assert plain_mean > 0.02
    assert smoothed_mean < plain_mean / 5.0


def test_ema_still_passes_gradient_through_the_current_batch():
    """Straight-through: value comes from the EMA, gradient from this batch."""
    reg = MinStateSize(l2=1.0, min_share=0.2, ema_decay=0.9)
    reg(tf.constant(np.full((4, 3), 1.0 / 3.0, dtype=np.float32)))  # seed the EMA

    x = tf.Variable(np.array([[0.97, 0.02, 0.01]] * 4, dtype=np.float32))
    with tf.GradientTape() as tape:
        penalty = reg(x)
    grad = tape.gradient(penalty, x).numpy()

    assert np.all(np.isfinite(grad))
    assert np.any(grad != 0.0)


def test_ema_applies_to_state_size_entropy_too():
    """Smoothing lives on the shared base class, not just MinStateSize."""
    reg = StateSizeEntropy(l2=1.0, ema_decay=0.9)
    assert reg.get_config()["ema_decay"] == 0.9
    for batch in _compliant_batches(3, 64):
        assert np.isfinite(float(reg(batch)))
    assert float(reg._smoothers["state_sizes"].step) == 3.0


def test_ema_decay_validated():
    """ema_decay outside [0, 1) is rejected at construction."""
    for bad in (-0.1, 1.0, 1.5):
        with pytest.raises(ValueError, match="ema_decay"):
            MinStateSize(l2=1.0, ema_decay=bad)


# -------- Tests for scheduled regularizer scalars --------


def test_scheduled_l2_is_read_at_call_time():
    """A ScheduledValue l2 must change the penalty without reconstructing."""
    l2 = ScheduledValue(start=0.0, end=2.0, duration_epochs=1)
    reg = MinStateSize(l2=l2, min_share=0.2)
    x = _marginal_weights([0.9, 0.05, 0.05])

    # Unscheduled, the value sits at `end` -- the fully enforced penalty.
    baseline = MinStateSize(l2=2.0, min_share=0.2)(x).numpy()
    assert baseline > 0.0
    np.testing.assert_allclose(reg(x).numpy(), baseline, rtol=1e-6)

    l2.assign_epoch(0)
    np.testing.assert_allclose(reg(x).numpy(), 0.0, atol=1e-7)
    l2.assign_epoch(1)
    np.testing.assert_allclose(reg(x).numpy(), baseline, rtol=1e-6)


def test_scheduled_degrees_of_freedom_target():
    """Annealing the DoF target is the recommended form for that penalty."""
    target = ScheduledValue(start=9.0, end=3.0, duration_epochs=6)
    reg = DegreesOfFreedom(l2=1.0, target=target)
    x = _hard_assignment_weights(n_states=9, n_alive=3)

    # df is 3 here. Unscheduled the target sits at `end` (3), so no penalty.
    np.testing.assert_allclose(reg(x).numpy(), 0.0, atol=1e-4)
    target.assign_epoch(0)
    assert reg(x).numpy() > 30.0
    target.assign_epoch(6)
    np.testing.assert_allclose(reg(x).numpy(), 0.0, atol=1e-4)


def test_scheduled_degrees_of_freedom_target_validated_over_whole_schedule():
    """A target dipping below 1 anywhere in the schedule is rejected."""
    with pytest.raises(AssertionError, match="whole"):
        DegreesOfFreedom(
            l2=1.0, target=ScheduledValue(start=9.0, end=0.5, duration_epochs=4)
        )


def test_scheduled_values_survive_get_config():
    """Regularizer configs serialize ScheduledValues instead of coercing to float."""
    l2 = ScheduledValue(start=0.0, end=1.0, duration_epochs=5)
    config = DegreesOfFreedom(l2=l2, target=3.0).get_config()
    assert isinstance(config["l2"], dict)

    restored = DegreesOfFreedom.from_config(config)
    assert isinstance(restored.l2, ScheduledValue)
    assert restored.l2.get_config() == l2.get_config()
    assert restored.target == 3.0


# -------- Smoothing and scheduling on the row-entropy and trace penalties --------


def test_ema_smooths_mean_row_entropy():
    """TargetEntropy averages a row statistic, which is also convex in the mean."""
    reg = Uniform(l2=1.0, ema_decay=0.8)
    for batch in _compliant_batches(4, 64):
        assert np.isfinite(float(reg(batch)))
    assert float(reg._smoothers["mean_row_entropy"].step) == 4.0
    assert reg.get_config()["ema_decay"] == 0.8


def test_ema_smooths_the_trace_ratio_termwise():
    """DegreesOfFreedom must smooth numerator and denominator, not the ratio.

    The kernel trace is a ratio estimator, so averaging the ratio itself would
    retain the small-batch bias that smoothing is meant to remove.
    """
    reg = DegreesOfFreedom(l2=1.0, target=3.0, ema_decay=0.9)
    for batch in _compliant_batches(3, 64):
        assert np.isfinite(float(reg(batch)))
    assert set(reg._smoothers) == {"trace_numerator", "trace_denominator"}
    assert float(reg._smoothers["trace_numerator"].step) == 3.0


def test_ema_reduces_trace_variance_on_small_batches():
    """Smoothing must actually stabilize the trace estimate across batches."""
    plain = DegreesOfFreedom(l2=1.0, target=3.0)
    smoothed = DegreesOfFreedom(l2=1.0, target=3.0, ema_decay=0.95)

    plain_vals, smoothed_vals = [], []
    for batch in _compliant_batches(200, 32):
        plain_vals.append(float(plain(batch)))
        smoothed_vals.append(float(smoothed(batch)))

    assert np.std(smoothed_vals[100:]) < np.std(plain_vals[100:]) / 3.0


def test_entropy_fraction_is_the_k_independent_knob():
    """entropy_fraction=f must equal target=log(f * K) for the K at hand."""
    n_states = 8
    x = _hard_assignment_weights(n_states=n_states, n_alive=3)
    by_fraction = TargetEntropy(l2=1.0, entropy_fraction=0.5)(x).numpy()
    by_target = TargetEntropy(l2=1.0, target=float(np.log(0.5 * n_states)))(x).numpy()
    np.testing.assert_allclose(by_fraction, by_target, rtol=1e-6)


def test_entropy_fraction_and_target_are_mutually_exclusive():
    """Passing both is ambiguous and rejected."""
    with pytest.raises(ValueError, match="not both"):
        TargetEntropy(l2=1.0, target=1.0, entropy_fraction=0.5)


def test_entropy_fraction_validated_over_a_whole_schedule():
    """Both endpoints of a scheduled fraction must be in (0, 1]."""
    with pytest.raises(ValueError, match="entropy_fraction"):
        TargetEntropy(
            l2=1.0,
            entropy_fraction=ScheduledValue(start=1.5, end=0.5, duration_epochs=4),
        )


def test_scheduled_entropy_fraction_starts_at_the_initialization():
    """Scheduling from 1.0 starts the target where uniform weights already are."""
    fraction = ScheduledValue(start=1.0, end=0.5, duration_epochs=4)
    reg = TargetEntropy(l2=1.0, entropy_fraction=fraction)
    n_states = 8
    uniform = tf.constant(np.full((4, n_states), 1.0 / n_states, dtype=np.float32))

    # At epoch 0 the target is log(K), exactly the entropy of uniform weights.
    fraction.assign_epoch(0)
    np.testing.assert_allclose(reg(uniform).numpy(), 0.0, atol=1e-6)
    # Once tightened, the same uniform weights are penalized.
    fraction.assign_epoch(4)
    assert reg(uniform).numpy() > 0.0


def test_scheduled_min_share_validates_the_strictest_value():
    """Feasibility is checked against the strictest floor the schedule reaches."""
    reg = MinStateSize(
        l2=1.0, min_share=ScheduledValue(start=0.01, end=0.25, duration_epochs=4)
    )
    with pytest.raises(ValueError, match="not satisfiable"):
        reg(_marginal_weights([0.2] * 5))


def test_scheduled_active_fraction_round_trips():
    """A scheduled active_fraction serializes as a nested config."""
    fraction = ScheduledValue(start=1.0, end=0.5, duration_epochs=4)
    config = StateSizeEntropy(l2=0.1, active_fraction=fraction).get_config()
    assert isinstance(config["active_fraction"], dict)
    restored = StateSizeEntropy.from_config(config)
    assert isinstance(restored._active_fraction, ScheduledValue)
    assert restored._active_fraction.get_config() == fraction.get_config()
