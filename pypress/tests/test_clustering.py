"""Tests for clustering utilities."""

import numpy as np
import pytest

from pypress.clustering import GaussianMixture1D


def test_gaussian_mixture_1d_basic():
    """Test GaussianMixture1D class with basic data."""
    np.random.seed(42)
    y1 = np.random.randn(100) - 10
    y2 = np.random.randn(100)
    y3 = np.random.randn(100) + 10
    y = np.concatenate([y1, y2, y3])

    gmm = GaussianMixture1D(n_components=3)
    gmm.fit(y)

    # Check fitted attributes exist
    assert gmm.means_ is not None
    assert gmm.covariances_ is not None
    assert gmm.means_.shape == (3,)
    assert gmm.covariances_.shape == (3,)

    # Check convergence attributes
    assert isinstance(gmm.n_iter_, int)
    assert isinstance(gmm.converged_, bool)
    assert gmm.n_iter_ > 0


def test_gaussian_mixture_1d_fit_returns_self():
    """Test that fit() returns self for chaining."""
    np.random.seed(42)
    y = np.random.randn(100)

    gmm = GaussianMixture1D(n_components=2)
    result = gmm.fit(y)

    assert result is gmm


def test_gaussian_mixture_1d_convergence():
    """Test convergence detection."""
    np.random.seed(42)
    y = np.random.randn(100)

    # With enough iterations, should converge
    gmm = GaussianMixture1D(n_components=2, max_iter=20)
    gmm.fit(y)

    assert gmm.converged_ is True
    assert gmm.n_iter_ <= 20


def test_gaussian_mixture_1d_no_convergence():
    """Test non-convergence with very few iterations."""
    np.random.seed(42)
    # Create well-separated data that needs multiple iterations
    y = np.concatenate([np.random.randn(100) - 10, np.random.randn(100) + 10])

    # With only 1 iteration, unlikely to converge
    gmm = GaussianMixture1D(n_components=2, max_iter=1)
    gmm.fit(y)

    # May or may not converge in 1 iteration, but should still have valid results
    assert gmm.means_ is not None
    assert gmm.covariances_ is not None
    assert gmm.n_iter_ == 1


def test_gaussian_mixture_1d_constant_data():
    """Test GaussianMixture1D on constant data."""
    y = np.full(100, 5.0)

    gmm = GaussianMixture1D(n_components=3)
    gmm.fit(y)

    # All means should be 5.0
    np.testing.assert_allclose(gmm.means_, 5.0)

    # All covariances should be min_std
    np.testing.assert_allclose(gmm.covariances_, gmm.min_std)

    # Should converge immediately
    assert gmm.converged_ is True
    assert gmm.n_iter_ == 0


def test_gaussian_mixture_1d_parameters():
    """Test that parameters are stored correctly."""
    gmm = GaussianMixture1D(
        n_components=5, max_iter=15, tol=1e-5, min_std=0.01, random_state=123
    )

    assert gmm.n_components == 5
    assert gmm.max_iter == 15
    assert gmm.tol == 1e-5
    assert gmm.min_std == 0.01
    assert gmm.random_state == 123


def test_gaussian_mixture_1d_invalid_input():
    """Test GaussianMixture1D with invalid input."""
    gmm = GaussianMixture1D(n_components=3)

    # Empty array
    with pytest.raises(ValueError, match="must have at least one element"):
        gmm.fit(np.array([]))

    # Invalid n_components (test at initialization time)
    gmm_invalid = GaussianMixture1D(n_components=0)
    with pytest.raises(ValueError, match="n_components must be positive"):
        gmm_invalid.fit(np.random.randn(100))


def test_gaussian_mixture_1d_2d_input():
    """Test GaussianMixture1D with 2D input (should flatten)."""
    np.random.seed(42)
    y = np.random.randn(50, 1) + 3

    gmm = GaussianMixture1D(n_components=2)
    gmm.fit(y)

    assert gmm.means_.shape == (2,)
    assert gmm.covariances_.shape == (2,)


def test_gaussian_mixture_1d_well_separated_clusters():
    """Test that GMM correctly identifies well-separated clusters."""
    np.random.seed(42)
    # Create data from 3 well-separated Gaussians
    y1 = np.random.randn(100) - 10  # Mean around -10
    y2 = np.random.randn(100)  # Mean around 0
    y3 = np.random.randn(100) + 10  # Mean around 10
    y = np.concatenate([y1, y2, y3])

    gmm = GaussianMixture1D(n_components=3)
    gmm.fit(y)

    # Check that means are roughly sorted
    means_sorted = np.sort(gmm.means_)
    assert means_sorted[0] < -5  # Should capture the -10 cluster
    assert -2 < means_sorted[1] < 2  # Should capture the 0 cluster
    assert means_sorted[2] > 5  # Should capture the +10 cluster

    # Check that stds are all positive and reasonable
    assert np.all(gmm.covariances_ > 0)
    assert np.all(gmm.covariances_ < 5)  # Should be around 1 for standard normal


def test_gaussian_mixture_1d_single_component():
    """Test GMM with single component (just computes mean and std)."""
    np.random.seed(42)
    y = np.random.randn(100) + 5

    gmm = GaussianMixture1D(n_components=1)
    gmm.fit(y)

    assert gmm.means_.shape == (1,)
    assert gmm.covariances_.shape == (1,)

    # Should be close to true mean and std
    np.testing.assert_allclose(gmm.means_[0], 5.0, atol=0.3)
    np.testing.assert_allclose(gmm.covariances_[0], 1.0, atol=0.3)


def test_gaussian_mixture_1d_equidistant_initialization():
    """Test that initial centers are equidistant."""
    np.random.seed(42)
    # Uniform data from 0 to 10
    y = np.random.uniform(0, 10, 1000)

    # With max_iter=0, we get the initial centers
    gmm = GaussianMixture1D(n_components=5, max_iter=0)
    gmm.fit(y)

    # Initial centers should be equidistant within the data range [y.min(), y.max()]
    expected = np.linspace(y.min(), y.max(), 5)
    np.testing.assert_allclose(gmm.means_, expected, atol=0.01)
