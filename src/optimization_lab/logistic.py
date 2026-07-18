"""Binary logistic regression trained without a machine-learning framework."""

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

    array = np.asarray(values, dtype=np.float64)
    result = np.empty_like(array)
    positive = array >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-array[positive]))
    negative_exp = np.exp(array[~positive])
    result[~positive] = negative_exp / (1.0 + negative_exp)
    return result


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
    label_values = np.asarray(labels, dtype=np.float64)
    probability_values = np.asarray(probabilities, dtype=np.float64)
    if label_values.shape != probability_values.shape or label_values.ndim != 1:
        raise ValueError("labels and probabilities must be matching vectors")
    clipped = np.clip(probability_values, 1e-12, 1.0 - 1e-12)
    return float(
        -np.mean(label_values * np.log(clipped) + (1.0 - label_values) * np.log(1.0 - clipped))
    )


def logistic_gradients(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    weights: np.ndarray,
    bias: float,
) -> tuple[np.ndarray, float]:
    feature_values, label_values = _validated_problem(features, labels)
    weight_values = np.asarray(weights, dtype=np.float64)
    if weight_values.shape != (feature_values.shape[1],):
        raise ValueError("weights must have one value per feature column")
    probabilities = sigmoid(feature_values @ weight_values + bias)
    errors = probabilities - label_values
    weight_gradient = feature_values.T @ errors / len(label_values)
    bias_gradient = float(np.mean(errors))
    return weight_gradient, bias_gradient


def fit_logistic_regression(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    learning_rate: float = 0.2,
    epochs: int = 500,
) -> LogisticFitResult:
    feature_values, label_values = _validated_problem(features, labels)
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if epochs < 1:
        raise ValueError("epochs must be at least 1")

    weights = np.zeros(feature_values.shape[1], dtype=np.float64)
    bias = 0.0
    losses: list[float] = []
    for epoch in range(epochs + 1):
        probabilities = sigmoid(feature_values @ weights + bias)
        losses.append(binary_cross_entropy(label_values, probabilities))
        if epoch == epochs:
            break
        weight_gradient, bias_gradient = logistic_gradients(
            feature_values,
            label_values,
            weights=weights,
            bias=bias,
        )
        weights -= learning_rate * weight_gradient
        bias -= learning_rate * bias_gradient
    return LogisticFitResult(weights=weights, bias=bias, losses=tuple(losses))

