"""One-feature linear regression trained with batch gradient descent."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LinearFitResult:
    slope: float
    intercept: float
    losses: tuple[float, ...]
    diverged: bool

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.slope * np.asarray(x, dtype=np.float64) + self.intercept


def _validated_vectors(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    if x_values.ndim != 1 or y_values.ndim != 1:
        raise ValueError("x and y must be one-dimensional")
    if len(x_values) == 0 or len(x_values) != len(y_values):
        raise ValueError("x and y must have the same non-zero length")
    if not np.all(np.isfinite(x_values)) or not np.all(np.isfinite(y_values)):
        raise ValueError("x and y must contain only finite values")
    return x_values, y_values


def mean_squared_error(y_true: np.ndarray, y_predicted: np.ndarray) -> float:
    true_values, predicted_values = _validated_vectors(y_true, y_predicted)
    residuals = predicted_values - true_values
    return float(np.mean(residuals * residuals))


def linear_gradients(
    x: np.ndarray,
    y: np.ndarray,
    *,
    slope: float,
    intercept: float,
) -> tuple[float, float]:
    """Return dMSE/dslope and dMSE/dintercept."""

    x_values, y_values = _validated_vectors(x, y)
    residuals = slope * x_values + intercept - y_values
    slope_gradient = 2.0 * float(np.mean(residuals * x_values))
    intercept_gradient = 2.0 * float(np.mean(residuals))
    return slope_gradient, intercept_gradient


def fit_linear_regression(
    x: np.ndarray,
    y: np.ndarray,
    *,
    learning_rate: float = 0.08,
    epochs: int = 500,
    initial_slope: float = 0.0,
    initial_intercept: float = 0.0,
) -> LinearFitResult:
    """Fit slope and intercept while retaining the full loss history."""

    x_values, y_values = _validated_vectors(x, y)
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if epochs < 1:
        raise ValueError("epochs must be at least 1")

    slope = float(initial_slope)
    intercept = float(initial_intercept)
    losses: list[float] = []
    diverged = False

    with np.errstate(over="ignore", invalid="ignore"):
        for epoch in range(epochs + 1):
            predictions = slope * x_values + intercept
            loss = mean_squared_error(y_values, predictions)
            losses.append(loss)
            if not np.isfinite(loss):
                diverged = True
                break
            if epoch == epochs:
                break
            slope_gradient, intercept_gradient = linear_gradients(
                x_values,
                y_values,
                slope=slope,
                intercept=intercept,
            )
            slope -= learning_rate * slope_gradient
            intercept -= learning_rate * intercept_gradient
            if not np.isfinite(slope) or not np.isfinite(intercept):
                diverged = True
                break

    return LinearFitResult(
        slope=slope,
        intercept=intercept,
        losses=tuple(losses),
        diverged=diverged,
    )

