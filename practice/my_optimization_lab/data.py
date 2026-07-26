"""Deterministic datasets whose true generating process is known."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LinearDataset:
    x: np.ndarray
    y: np.ndarray
    true_slope: float
    true_intercept: float


@dataclass(frozen=True)
class BinaryDataset:
    features: np.ndarray
    labels: np.ndarray


def make_linear_dataset(
    *,
    sample_count: int = 160,
    slope: float = 3.5,
    intercept: float = -1.25,
    noise_sigma: float = 0.12,
    seed: int = 7,
) -> LinearDataset:
    """Generate y = slope*x + intercept + Gaussian noise."""

    if sample_count < 2:
        raise ValueError("sample_count must be at least 2")
    if noise_sigma < 0:
        raise ValueError("noise_sigma cannot be negative")
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.0, 2.0, size=sample_count)
    noise = rng.normal(0.0, noise_sigma, size=sample_count)
    y = slope * x + intercept + noise
    return LinearDataset(x=x, y=y, true_slope=slope, true_intercept=intercept)


def make_binary_dataset(
    *,
    samples_per_class: int = 100,
    spread: float = 0.48,
    seed: int = 11,
) -> BinaryDataset:
    """Generate two separable two-dimensional Gaussian clusters."""

    if samples_per_class < 2:
        raise ValueError("samples_per_class must be at least 2")
    if spread <= 0:
        raise ValueError("spread must be positive")
    rng = np.random.default_rng(seed)
    negative = rng.normal(loc=(-1.35, -1.0), scale=spread, size=(samples_per_class, 2))
    positive = rng.normal(loc=(1.35, 1.0), scale=spread, size=(samples_per_class, 2))
    features = np.vstack((negative, positive))
    labels = np.concatenate(
        (np.zeros(samples_per_class, dtype=np.float64), np.ones(samples_per_class, dtype=np.float64))
    )
    order = rng.permutation(len(labels))
    return BinaryDataset(features=features[order], labels=labels[order])

