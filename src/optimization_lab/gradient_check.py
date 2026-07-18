"""Finite-difference utilities for checking hand-derived gradients."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def finite_difference_gradient(
    objective: Callable[[np.ndarray], float],
    parameters: np.ndarray,
    *,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Approximate each partial derivative with a centered finite difference."""

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    values = np.asarray(parameters, dtype=np.float64)
    result = np.zeros_like(values)
    for index in np.ndindex(values.shape):
        positive = values.copy()
        negative = values.copy()
        positive[index] += epsilon
        negative[index] -= epsilon
        result[index] = (objective(positive) - objective(negative)) / (2.0 * epsilon)
    return result


def relative_gradient_error(analytic: np.ndarray, numerical: np.ndarray) -> float:
    """Return a scale-aware error between two gradient vectors."""

    analytic_values = np.asarray(analytic, dtype=np.float64)
    numerical_values = np.asarray(numerical, dtype=np.float64)
    if analytic_values.shape != numerical_values.shape:
        raise ValueError("gradients must have identical shapes")
    denominator = max(
        1e-12,
        float(np.linalg.norm(analytic_values) + np.linalg.norm(numerical_values)),
    )
    return float(np.linalg.norm(analytic_values - numerical_values) / denominator)

