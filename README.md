# pypress: Predictive State Smoothing (PRESS) in Python (`tf.keras`)

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)
[![GPLv3 license](https://img.shields.io/badge/License-GPLv3-blue.svg)](http://perso.crans.org/besson/LICENSE.html)

Predictive State Smoothing (PRESS) is a semi-parametric statistical machine learning algorithm
for regression and classification problems.

`pypress` is built on top of TensorFlow Keras and implements predictive learning algorithms proposed in


* Goerg (2018) *[Classification using Predictive State Smoothing (PRESS): A scalable kernel classifier for high-dimensional features with variable selection](https://research.google/pubs/pub46767/)*.

* Goerg (2017) *[Predictive State Smoothing (PRESS): Scalable non-parametric regression for high-dimensional data with variable selection](https://research.google/pubs/pub46141/).*


# Installation

It can be installed directly from `github.com` using:
```
pip install git+https://github.com/gmgeorg/pypress.git
```


# Example usage

PRESS is available as 2 layers that need to be added one after the other; alternatively
there is a `PRESS()` wrapper feed-forward layer that applies both layers at once.


```python
import tensorflow as tf
from pypress import layers
from pypress import regularizers

mod = tf.keras.Sequential()
# see layers.PRESS() for single layer wrapper
mod.add(layers.PredictiveStateSimplex(
            n_states=6,
            activity_regularizer=regularizers.Uniform(0.01),
            input_dim=X.shape[1]))
mod.add(layers.PredictiveStateMeans(units=1, activation="sigmoid"))
mod.compile(loss="binary_crossentropy",
            optimizer=tf.keras.optimizers.Nadam(learning_rate=0.01),
            metrics=[tf.keras.metrics.AUC(curve="PR", name="auc_pr")])
mod.summary()
```

```
Model: "sequential_12"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 predictive_states_simplex_1  (None, 6)                186
 1 (PredictiveStatesSimplex)

 predictive_state_means_11 (  (None, 1)                6
 PredictiveStateMeans)

=================================================================
Total params: 192
Trainable params: 192
Non-trainable params: 0
```


See also the [`notebook/demo.ipynb`](notebooks/demo.ipynb) for end to end examples for PRESS regression and classification models.
