"""One-feature linear regression trained with batch gradient descent.

PRACTICE STUBS. You implement the three functions marked TODO.
The dataclass and the `_validated_vectors` helper are given — reuse them.
Grade yourself from the practice/ folder with:
    PYTHONPATH=. python -m unittest test_practice -v
`git show HEAD:src/optimization_lab/linear.py` is the answer key if stuck.
"""

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
    


def linear_gradients(
    x: np.ndarray,
    y: np.ndarray,
    *,
    slope: float,
    intercept: float,
) -> tuple[float, float]:
    """Return dMSE/dslope and dMSE/dintercept."""

    # TODO 2  (the calculus)
    # L = mean( (slope*x + intercept - y) ** 2 )
    # residual = slope*x + intercept - y
    #   dL/dslope     = 2 * mean(residual * x)
    #   dL/dintercept = 2 * mean(residual)
    # Return (slope_gradient, intercept_gradient) as plain floats.
    raise NotImplementedError("implement linear_gradients")


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

    # TODO 4  (the loop — do this AFTER TODO 1-3 in gradient_check.py)
    # 1. validate; reject learning_rate <= 0 and epochs < 1 with ValueError.
    # 2. start slope/intercept from the initial_* args; losses = []; diverged = False.
    # 3. wrap the loop in:  with np.errstate(over="ignore", invalid="ignore"):
    # 4. for epoch in range(epochs + 1):
    #       predict -> compute loss -> losses.append(loss)
    #       if not np.isfinite(loss): diverged = True; break
    #       if epoch == epochs: break            # final pass only records the loss
    #       compute gradients, then:
    #         slope     -= learning_rate * slope_gradient
    #         intercept -= learning_rate * intercept_gradient
    #       if slope/intercept became non-finite: diverged = True; break
    # 5. return LinearFitResult(slope, intercept, tuple(losses), diverged)
    raise NotImplementedError("implement fit_linear_regression")
