"""A regularizer config from one backend builds the other, and round-trips."""

import json

import pytest
import tensorflow as tf

from pypress.keras import regularizers as keras_regularizers
from pypress.torch import regularizers as torch_regularizers

from .helpers import Scheduled, REGULARIZER_SETTINGS, build, setting_id

_SETTINGS = pytest.mark.parametrize(
    "setting", REGULARIZER_SETTINGS, ids=[setting_id(s) for s in REGULARIZER_SETTINGS]
)


@_SETTINGS
def test_get_config_is_identical(setting):
    """The same constructor arguments give the same `get_config()`."""
    name, kwargs = setting
    keras_reg = build("keras", name, kwargs)
    torch_reg = build("torch", name, kwargs)
    assert torch_reg.get_config() == keras_reg.get_config()
    json.dumps(torch_reg.get_config())


@_SETTINGS
def test_flat_keras_config_round_trips_through_torch(setting):
    """Keras config -> torch class -> config equals the original."""
    name, kwargs = setting
    keras_config = build("keras", name, kwargs).get_config()
    torch_reg = torch_regularizers.from_config(keras_config | {"class_name": name})
    assert type(torch_reg).__name__ == name
    assert torch_reg.get_config() == keras_config


@_SETTINGS
def test_serialized_form_is_identical_and_loads_in_both(setting):
    """Serialized regularizers are equal, and each backend reads the other's."""
    name, kwargs = setting
    keras_reg = build("keras", name, kwargs)
    torch_reg = build("torch", name, kwargs)
    keras_form = tf.keras.regularizers.serialize(keras_reg)
    assert torch_regularizers.serialize(torch_reg) == keras_form

    from_keras = torch_regularizers.from_config(keras_form)
    assert torch_regularizers.serialize(from_keras) == keras_form
    from_torch = tf.keras.regularizers.deserialize(
        torch_regularizers.serialize(torch_reg)
    )
    assert tf.keras.regularizers.serialize(from_torch) == keras_form


def test_combined_regularizer_config_round_trips_across_backends():
    """The composite serializes its children in the Keras form."""
    children = [
        ("Uniform", dict(l2=0.5)),
        (
            "DegreesOfFreedom",
            dict(
                l2=1.0,
                target=Scheduled(start=5.0, end=3.0, duration_epochs=4),
                ema_decay=0.9,
                normalize=True,
            ),
        ),
    ]
    keras_reg = keras_regularizers.CombinedRegularizer(
        [build("keras", n, k) for n, k in children]
    )
    torch_reg = torch_regularizers.CombinedRegularizer(
        [build("torch", n, k) for n, k in children]
    )
    keras_form = tf.keras.regularizers.serialize(keras_reg)
    assert torch_regularizers.serialize(torch_reg) == keras_form
    restored = torch_regularizers.from_config(keras_form)
    assert torch_regularizers.serialize(restored) == keras_form
    assert (
        tf.keras.regularizers.serialize(
            tf.keras.regularizers.deserialize(torch_regularizers.serialize(restored))
        )
        == keras_form
    )


def test_combined_regularizer_penalties_match():
    """Summing children gives equal penalties in both backends."""
    from .data import BATCHES
    from .helpers import assert_penalties_equal

    children = [
        ("MinStateSize", dict(l2=2.0, min_share=0.05)),
        ("DegreesOfFreedom", dict(l2=1.0, target=3.0)),
        ("StateSizeEntropy", dict(l2=0.5, active_fraction=0.8)),
    ]
    keras_reg = keras_regularizers.CombinedRegularizer(
        [build("keras", n, k) for n, k in children]
    )
    torch_reg = torch_regularizers.CombinedRegularizer(
        [build("torch", n, k) for n, k in children]
    )
    assert_penalties_equal(keras_reg, torch_reg, BATCHES["random_k9"])
