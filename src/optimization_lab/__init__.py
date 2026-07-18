"""Small, inspectable optimization algorithms for learning ML fundamentals."""

from .linear import LinearFitResult, fit_linear_regression, linear_gradients, mean_squared_error
from .logistic import LogisticFitResult, fit_logistic_regression, logistic_gradients

__all__ = [
    "LinearFitResult",
    "LogisticFitResult",
    "fit_linear_regression",
    "fit_logistic_regression",
    "linear_gradients",
    "logistic_gradients",
    "mean_squared_error",
]

