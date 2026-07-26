"""Binary logistic regression trained without a machine-learning framework.

PRACTICE STUBS. Implement the four TODO functions. The dataclass and the
`_validated_problem` helper are given — reuse them. Notice how the gradient
mirrors linear regression: error = prediction - truth, again.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LogisticFitResult:
    weights: np.ndarray
    bias: float
    losses: tuple[float, ...]

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.weights):
            raise ValueError("features must be a matrix with one column per weight")
        return sigmoid(values @ self.weights + self.bias)

    def predict(self, features: np.ndarray, *, threshold: float = 0.5) -> np.ndarray:
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must lie between 0 and 1")
        return (self.predict_proba(features) >= threshold).astype(np.int64)


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid."""

    # TODO 5
    # sigma(z) = 1 / (1 + e^{-z}), but split by sign to avoid overflow:
    #   for z >= 0:  1 / (1 + exp(-z))
    #   for z <  0:  exp(z) / (1 + exp(z))     # same value, exp(z) <= 1 so it is safe
    # Build an empty result, mask positive = array >= 0, fill each branch.
    raise NotImplementedError("implement sigmoid")


def _validated_problem(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    feature_values = np.asarray(features, dtype=np.float64)
    label_values = np.asarray(labels, dtype=np.float64)
    if feature_values.ndim != 2 or feature_values.shape[0] == 0:
        raise ValueError("features must be a non-empty matrix")
    if label_values.ndim != 1 or len(label_values) != feature_values.shape[0]:
        raise ValueError("labels must have one value per feature row")
    if not np.all(np.isfinite(feature_values)) or not np.all(np.isfinite(label_values)):
        raise ValueError("features and labels must be finite")
    if not np.all(np.isin(label_values, (0.0, 1.0))):
        raise ValueError("labels must be binary values 0 or 1")
    return feature_values, label_values


def binary_cross_entropy(labels: np.ndarray, probabilities: np.ndarray) -> float:
    # TODO 6
    # clip probabilities to [1e-12, 1 - 1e-12] so log never sees 0 or 1.
    #   L = -mean( y*log(p) + (1-y)*log(1-p) )
    # Return a plain float. (labels and probabilities are matching 1-D vectors.)
    raise NotImplementedError("implement binary_cross_entropy")


def logistic_gradients(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    weights: np.ndarray,
    bias: float,
) -> tuple[np.ndarray, float]:
    # TODO 7  (the calculus — same shape as linear!)
    # Use _validated_problem(features, labels). weights must be shape (n_features,).
    #   z = features @ weights + bias
    #   p = sigmoid(z)
    #   error = p - labels                       # the "residual" again
    #   weight_gradient = features.T @ error / n_samples
    #   bias_gradient   = mean(error)
    # Return (weight_gradient as np.ndarray, bias_gradient as float).
    raise NotImplementedError("implement logistic_gradients")


def fit_logistic_regression(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    learning_rate: float = 0.2,
    epochs: int = 500,
) -> LogisticFitResult:
    # TODO 8  (the loop — same shape as fit_linear_regression, no divergence guard needed)
    # 1. validate; reject learning_rate <= 0 and epochs < 1.
    # 2. weights = zeros(n_features); bias = 0.0; losses = [].
    # 3. for epoch in range(epochs + 1):
    #       p = sigmoid(features @ weights + bias)
    #       losses.append(binary_cross_entropy(labels, p))
    #       if epoch == epochs: break
    #       compute gradients; weights -= lr * weight_grad; bias -= lr * bias_grad
    # 4. return LogisticFitResult(weights, bias, tuple(losses))
    raise NotImplementedError("implement fit_logistic_regression")
