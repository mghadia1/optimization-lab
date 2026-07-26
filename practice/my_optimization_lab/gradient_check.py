"""Finite-difference utilities for checking hand-derived gradients.

PRACTICE STUBS. Implement both functions. Do these right after TODO 2
(linear_gradients) — together they let you PROVE your calculus is correct.
"""

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

    # TODO 3a
    # Reject epsilon <= 0 with ValueError.
    # values = float64 copy of parameters; result = zeros_like(values).
    # For each index in np.ndindex(values.shape):
    #   make a +eps copy and a -eps copy of values (only that one entry changes),
    #   result[index] = (objective(plus) - objective(minus)) / (2 * epsilon)
    # Return result. (This is the DEFINITION of a derivative, per parameter.)
    raise NotImplementedError("implement finite_difference_gradient")


def relative_gradient_error(analytic: np.ndarray, numerical: np.ndarray) -> float:
    """Return a scale-aware error between two gradient vectors."""

    # TODO 3b
    # Require identical shapes (ValueError otherwise).
    # denominator = max(1e-12, norm(analytic) + norm(numerical))
    # return norm(analytic - numerical) / denominator     (use np.linalg.norm)
    raise NotImplementedError("implement relative_gradient_error")
