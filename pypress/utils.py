"""Utility functions."""

from typing import Union, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

_STATE_COL = "state"


def state_size(
    weights: Union[pd.DataFrame, np.ndarray], normalize: bool = False
) -> Union[pd.Series, np.ndarray]:
    """Computes size of states as total sum of probabilities per state."""
    col_sums = weights.sum(axis=0)
    if isinstance(weights, pd.DataFrame):
        col_sums.index.name = _STATE_COL

    if normalize:
        col_sums /= col_sums.sum()
    return col_sums


def tf_state_size(weights: tf.Tensor) -> tf.Tensor:
    """Computes size of states for input Tensor."""
    return tf.reduce_sum(weights, axis=0)


def col_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Colum normalizes DataFrame (usually 'weights')."""
    return df.divide(state_size(df), axis=1)


def tf_col_normalize(x: tf.Tensor) -> tf.Tensor:
    """Computes column normalized tensor."""
    return tf.divide(x, tf_state_size(x))


def kernel_matrix(
    weights: Union[pd.DataFrame, np.ndarray],
) -> Union[pd.DataFrame, np.ndarray]:
    """Computes the kernel matrix implied by PRESS weights.

    The kernel matrix K is computed as:
        K = weights @ col_normalize(weights).T

    where col_normalize normalizes each column (state) to sum to 1.

    Args:
        weights: Weight matrix of shape (n_samples, n_states) representing
            predictive state probabilities.

    Returns:
        Kernel matrix of shape (n_samples, n_samples). If input is a DataFrame,
        returns a DataFrame with matching index. Otherwise returns a numpy array.

    Example:
        >>> import numpy as np
        >>> import pandas as pd
        >>> weights = pd.DataFrame({'s1': [0.1, 0.2, 0.7], 's2': [0.8, 0.1, 0.1]})
        >>> K = kernel_matrix(weights)
        >>> K.shape
        (3, 3)
    """
    if isinstance(weights, pd.DataFrame):
        # Use pandas operations to preserve index
        norm_weights = col_normalize(weights)
        kernel = weights.dot(norm_weights.transpose())
        return kernel
    else:
        # Use numpy operations
        weights_np = np.asarray(weights)
        col_sums = weights_np.sum(axis=0, keepdims=True)
        norm_weights = weights_np / col_sums
        kernel = weights_np @ norm_weights.T
        return kernel


def tr_kernel(weights: tf.Tensor) -> tf.Tensor:
    """Computes trace of kernel matrix implied by PRESS tensor."""
    return tf.reduce_sum(
        tf.linalg.diag_part(tf.matmul(tf.transpose(tf_col_normalize(weights)), weights))
    )


def agg_data_by_state(
    data: pd.DataFrame, weights: pd.DataFrame, agg_fn: str = "mean"
) -> pd.DataFrame:
    """Aggregates data by state. Will only aggregate numeric features."""
    if isinstance(data, pd.Series):
        data = data.to_frame()
    df_numeric = data.select_dtypes("number")
    if agg_fn == "mean":
        agg_state = col_normalize(weights).transpose().dot(df_numeric).transpose()
        agg_state.columns.name = None
        return agg_state

    raise NotImplementedError("agg_fn='%s' not supported", agg_fn)


def initialize_from_y(
    y: Union[np.ndarray, pd.Series],
    n_states: int,
    return_params: bool = False,
    max_iter: int = 10,
) -> Union[np.ndarray, Tuple[list, list]]:
    """Initialize PRESS layers from Gaussian Mixture Model fitted to training outputs.

    Fits a 1D Gaussian Mixture Model to the training outputs y using a simple
    k-means-style EM algorithm. Returns initialization values that can be directly
    passed to PredictiveStateMeans or PredictiveStateParams initializers.

    Args:
        y: Training outputs of shape (n_samples,) or (n_samples, n_outputs).
            For multi-output case (n_outputs > 1), returns the overall mean across
            all outputs.
        n_states: Number of predictive states (mixture components).
        return_params: If False, returns mean values suitable for PredictiveStateMeans.
            If True, returns (means, stds) suitable for PredictiveStateParams with
            Gaussian parameterization.
        max_iter: Maximum number of EM iterations for GMM fitting. Default is 10.

    Returns:
        If return_params=False:
            np.ndarray of shape (units, n_states) with state-conditional means,
            suitable for init_values in PredictiveStateMeans.
        If return_params=True:
            np.ndarray of shape (n_params_per_state, n_states) where
            row 0 is means and row 1 is stds, suitable for init_values in
            PredictiveStateParams with activations=["linear", "softplus"].

    Example:
        >>> # For PredictiveStateMeans
        >>> y_train = np.random.randn(1000, 2)  # 1000 samples, 2 outputs
        >>> init_mean = initialize_from_y(y_train, n_states=5)
        >>> layer = PredictiveStateMeans(units=2, init_values=init_mean)
        >>>
        >>> # For PredictiveStateParams with Gaussian parameterization
        >>> init_params = initialize_from_y(y_train, n_states=5, return_params=True)
        >>> layer = PredictiveStateParams(
        ...     n_params_per_state=2,
        ...     activations=["linear", "softplus"],
        ...     init_values=init_params
        ... )

    Note:
        This function uses a lightweight GMM implementation in clustering.py that
        does not require scikit-learn or tensorflow-probability. For 1D outputs,
        it fits the GMM directly. For multi-dimensional outputs, it computes the
        overall mean and std across all dimensions as a simple heuristic.
    """
    from .clustering import GaussianMixture1D

    # Convert to numpy array
    if isinstance(y, pd.Series):
        y = y.values

    y = np.asarray(y)

    # Handle shape
    if len(y.shape) == 1:
        # Shape (n_samples,) -> reshape to (n_samples, 1)
        y = y[:, np.newaxis]
        units = 1
    else:
        # Shape (n_samples, n_outputs)
        units = y.shape[1]

    # For multi-output, flatten to 1D for GMM fitting (simple heuristic)
    y_flat = y.ravel()

    # Fit GMM using scikit-learn-style API
    gmm = GaussianMixture1D(n_components=n_states, max_iter=max_iter)
    gmm.fit(y_flat)
    cluster_means = gmm.means_  # Shape: (n_states,)
    cluster_stds = gmm.covariances_  # Shape: (n_states,)

    if return_params:
        # Return format for PredictiveStateParams: (n_params_per_state, n_states) array
        # Row 0: means, Row 1: stds
        init_values = np.vstack([cluster_means, cluster_stds])  # Shape: (2, n_states)
        return init_values
    else:
        # Return format for PredictiveStateMeans: (units, n_states) array
        # Each row represents one output dimension, columns are states
        # For simplicity, use the same cluster means for all output dimensions
        init_values = np.tile(cluster_means, (units, 1))  # Shape: (units, n_states)
        return init_values
