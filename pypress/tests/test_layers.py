"""Tests for layers."""

from typing import Tuple

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from ..keras import layers


def _test_data(n_samples: int) -> Tuple[pd.DataFrame, pd.Series]:
    np.random.seed(10)
    feats = pd.DataFrame(np.random.normal(size=(n_samples, 7)))
    feats.columns = ["x" + str(k + 1) for k in range(feats.shape[1])]

    eps = np.random.normal(size=feats.shape[0])
    y = feats["x1"] + 5 * feats["x3"] - np.exp(feats["x5"]) + eps

    return feats, y


@pytest.mark.parametrize("n_states", [(1), (2), (5)])
def test_predictive_state_simplex(n_states):
    test_df, _ = _test_data(n_samples=3)
    pss_layer = layers.PredictiveStateSimplex(n_states=n_states)
    pred_states = pss_layer(test_df.values).numpy()
    assert pred_states.shape[0] == test_df.shape[0]
    assert pred_states.shape[1] == n_states

    row_sums = pred_states.sum(axis=1)
    np.testing.assert_allclose(row_sums, np.ones_like(row_sums))


@pytest.mark.parametrize("n_states,units", [(1, 1), (2, 4), (5, 10)])
def test_predictive_state_means(n_states, units):
    test_df, _ = _test_data(n_samples=3)
    pss_layer = layers.PredictiveStateSimplex(n_states=n_states)
    pred_states = pss_layer(test_df.values).numpy()
    cond_means = layers.PredictiveStateMeans(units=units)
    preds = cond_means(pred_states)
    assert preds.shape[0] == test_df.shape[0]
    assert preds.shape[1] == units


def test_use_in_model_works():
    feats, y = _test_data(n_samples=1000)

    model = tf.keras.Sequential()
    model.add(layers.PredictiveStateSimplex(5, input_dim=feats.shape[1]))
    model.add(layers.PredictiveStateMeans(1, "linear"))
    model.compile(loss="mse", optimizer=tf.keras.optimizers.Nadam(learning_rate=0.01))

    model.fit(feats, y, epochs=3)
    preds = model.predict(feats).ravel()

    cor_mat = np.corrcoef(preds, y)
    print(cor_mat)

    assert cor_mat[0, 1] > 0.88


def test_press_in_model_works():
    feats, y = _test_data(n_samples=1000)

    model = tf.keras.Sequential()
    model.add(layers.PRESS(units=1, n_states=5, input_dim=feats.shape[1]))
    model.compile(loss="mse", optimizer=tf.keras.optimizers.Nadam(learning_rate=0.01))

    model.fit(feats, y, epochs=3)
    preds = model.predict(feats).ravel()

    cor_mat = np.corrcoef(preds, y)
    print(cor_mat)

    assert cor_mat[0, 1] > 0.88
