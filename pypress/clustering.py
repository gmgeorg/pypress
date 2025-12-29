"""Clustering utilities for initializing PRESS layers."""

from typing import Optional

import numpy as np


class GaussianMixture1D:
    """1D Gaussian Mixture Model estimator.

    Fits a Gaussian Mixture Model to 1D data using a simple k-means-style EM
    algorithm. This is a lightweight implementation following the scikit-learn
    Estimator API pattern.

    Parameters
    ----------
    n_components : int
        Number of mixture components (clusters).
    max_iter : int, default=10
        Maximum number of EM iterations.
    tol : float, default=1e-6
        Convergence tolerance for cluster centers.
    min_std : float, default=1e-3
        Minimum standard deviation to prevent numerical issues.
    random_state : int, default=42
        Random seed for reproducibility (currently unused, reserved for future
        stochastic initialization options).

    Attributes
    ----------
    means_ : ndarray of shape (n_components,)
        Fitted means of mixture components.
    covariances_ : ndarray of shape (n_components,)
        Fitted standard deviations of mixture components.
    n_iter_ : int
        Number of EM iterations performed.
    converged_ : bool
        Whether the EM algorithm converged within max_iter.

    Examples
    --------
    >>> import numpy as np
    >>> from pypress.clustering import GaussianMixture1D
    >>> y = np.random.randn(1000)
    >>> gmm = GaussianMixture1D(n_components=3)
    >>> gmm.fit(y)
    >>> gmm.means_
    array([...])
    >>> gmm.covariances_
    array([...])

    Notes
    -----
    This implementation uses hard assignment (k-means style) rather than soft
    assignment (full EM) for simplicity and speed. For more sophisticated GMM
    fitting, consider using scikit-learn's GaussianMixture or TensorFlow Probability.
    """

    def __init__(
        self,
        n_components: int,
        max_iter: int = 10,
        tol: float = 1e-6,
        min_std: float = 1e-3,
        random_state: int = 42,
    ):
        """Initialize GaussianMixture1D estimator.

        Parameters
        ----------
        n_components : int
            Number of mixture components.
        max_iter : int, default=10
            Maximum number of EM iterations.
        tol : float, default=1e-6
            Convergence tolerance.
        min_std : float, default=1e-3
            Minimum standard deviation.
        random_state : int, default=42
            Random seed.
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.min_std = min_std
        self.random_state = random_state

        # Fitted attributes (set during fit())
        self.means_: Optional[np.ndarray] = None
        self.covariances_: Optional[np.ndarray] = None
        self.n_iter_: int = 0
        self.converged_: bool = False

    def fit(self, X: np.ndarray, y=None) -> "GaussianMixture1D":
        """Fit the Gaussian Mixture Model to data.

        Parameters
        ----------
        X : array-like of shape (n_samples,) or (n_samples, 1)
            Training data.
        y : None
            Ignored. Present for API compatibility.

        Returns
        -------
        self : GaussianMixture1D
            Fitted estimator.

        Raises
        ------
        ValueError
            If X is empty or n_components is invalid.
        """
        # Ensure X is 1D
        X = np.asarray(X).ravel()

        if len(X) == 0:
            raise ValueError("X must have at least one element")

        if self.n_components <= 0:
            raise ValueError(f"n_components must be positive, got {self.n_components}")

        # Initialize cluster centers on equidistant grid
        x_min, x_max = X.min(), X.max()

        # Handle edge case where all values are the same
        if x_min == x_max:
            self.means_ = np.full(self.n_components, x_min)
            self.covariances_ = np.full(self.n_components, self.min_std)
            self.n_iter_ = 0
            self.converged_ = True
            return self

        # Create equidistant grid for initial centers
        centers = np.linspace(x_min, x_max, self.n_components)

        # Initialize standard deviations (in case max_iter=0)
        stds = np.full(self.n_components, self.min_std)

        # Run k-means-style EM iterations
        self.converged_ = False
        for iteration in range(self.max_iter):
            # E-step: Assign each point to nearest center (hard assignment)
            distances = np.abs(X[:, np.newaxis] - centers[np.newaxis, :])
            assignments = np.argmin(distances, axis=1)

            # M-step: Update centers and compute standard deviations
            new_centers = np.zeros(self.n_components)

            for k in range(self.n_components):
                mask = assignments == k
                if mask.sum() > 0:
                    cluster_points = X[mask]
                    new_centers[k] = cluster_points.mean()
                    stds[k] = max(cluster_points.std(), self.min_std)
                else:
                    # If cluster is empty, keep previous center and use min_std
                    new_centers[k] = centers[k]
                    stds[k] = self.min_std

            # Check convergence
            if np.allclose(centers, new_centers, rtol=self.tol):
                self.converged_ = True
                centers = new_centers
                self.n_iter_ = iteration + 1
                break

            centers = new_centers
            self.n_iter_ = iteration + 1

        # Store fitted parameters
        self.means_ = centers
        self.covariances_ = stds

        return self
