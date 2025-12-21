# pypress: Predictive State Smoothing (PRESS) in Python (`tf.keras`)

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Github All Releases](https://img.shields.io/github/downloads/gmgeorg/pypress/total.svg)]()

Predictive State Smoothing (PRESS) is a semi-parametric statistical machine learning algorithm
for regression and classification problems. `pypress` is using TensorFlow Keras to implement
the predictive learning algorithms proposed in


* Goerg (2018) *[Classification using Predictive State Smoothing (PRESS): A scalable kernel classifier for high-dimensional features with variable selection](https://research.google/pubs/pub46767/)*.

* Goerg (2017) *[Predictive State Smoothing (PRESS): Scalable non-parametric regression for high-dimensional data with variable selection](https://research.google/pubs/pub46141/).*

See [below](#nutshell) for details on how PRESS works in a nutshell.


# Installation

It can be installed directly from `github.com` using:
```
pip install git+https://github.com/gmgeorg/pypress.git
```


# Example usage

PRESS is available as 2 layers that need to be added one after the other; alternatively
there is a `PRESS()` wrapper feed-forward layer that applies both layers at once.


```python
from sklearn.datasets import load_breast_cancer
import sklearn
X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_s = sklearn.preprocessing.robust_scale(X)  # See demo.ipynb to properly scale X with train/test split


import tensorflow as tf
from pypress.keras import layers
from pypress.keras import regularizers

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
mod.fit(X_s, y, epochs=10, validation_split=0.2)
```

```
Model: "sequential_12"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 predictive_state_simplex_1  (None, 6)                186
 1 (PredictiveStateSimplex)

 predictive_state_means_11 (  (None, 1)                6
 PredictiveStateMeans)

=================================================================
Total params: 192
Trainable params: 192
Non-trainable params: 0
```


See also the [`notebook/demo.ipynb`](notebooks/demo.ipynb) for end to end examples for PRESS regression and classification models.

# PRESS in a nutshell <a name="nutshell"/>

The figure below, adapted from **Goerg (2018)**, contrasts the architecture of a standard feed-forward Deep Neural Network (DNN) with the **Predictive State Smoothing (PRESS)** approach.

![PRESS architecture](imgs/press_architecture.png)

### 1. Standard Feed-Forward DNNs
In typical prediction problems, our goal is to model the conditional distribution $p(y \mid X)$ or the conditional expectation $E[y \mid X]$. A standard feed-forward network estimates this by directly mapping features ($X$) to an output through a series of highly non-linear transformations (as seen in Figure 3a).

### 2. The PRESS Decomposition
In contrast, PRESS decomposes the predictive distribution into a mixture distribution over **predictive states** ($S$). This architecture relies on a critical property: conditioned on a predictive state $j$, the output ($y$) becomes conditionally independent of the input features ($X$).

Mathematically, this is expressed as:

![PRESS equation](imgs/press_decomposition_equation.png)

The second equality holds because the state $j$ captures all relevant information from $X$ necessary to predict $y$, rendering the raw features redundant once the state is known.

### 3. Key Advantages and Clustering
The primary strength of this decomposition is that predictive states serve as **minimal sufficient statistics** for $y$. They provide an optimal informational summary—maximizing compression while retaining full predictive power.

An important byproduct of this framework is the ability to perform **predictive clustering**:
* Once the mapping from features ($X$) to the predictive state simplex is learned, observations can be clustered within the state space.
* Observations sharing similar predictive states are guaranteed to have similar predictive distributions for $y$, providing a principled way to group data based on future outcomes rather than raw input similarity.

### 4. Comparison to Mixture Density Networks (MDN)

While PRESS shares similarities with [Mixture Density Networks (MDN)](https://publications.aston.ac.uk/id/eprint/373/1/NCRG_94_004.pdf), there is a fundamental distinction. In an MDN, the output parameters are often direct functions of the features. In **PRESS**, the conditional independence of $y$ and $X$ given $S$ ensures that the output means are conditioned *only* on the predictive state, not the raw features.


## License

This project is licensed under the terms of the MIT license. See [LICENSE](https://github.com/gmgeorg/pypress/blob/main/LICENSE) for additional details.
