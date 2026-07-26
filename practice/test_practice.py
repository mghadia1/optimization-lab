from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from my_optimization_lab.cli import run_experiment
from my_optimization_lab.data import make_binary_dataset, make_linear_dataset
from my_optimization_lab.gradient_check import finite_difference_gradient, relative_gradient_error
from my_optimization_lab.linear import fit_linear_regression, linear_gradients, mean_squared_error
from my_optimization_lab.logistic import (
    binary_cross_entropy,
    fit_logistic_regression,
    logistic_gradients,
    sigmoid,
)


class LinearRegressionTests(unittest.TestCase):
    def test_analytic_gradient_matches_finite_difference(self) -> None:
        dataset = make_linear_dataset(seed=2)
        parameters = np.asarray((0.25, -0.4), dtype=np.float64)
        analytic = np.asarray(
            linear_gradients(
                dataset.x,
                dataset.y,
                slope=float(parameters[0]),
                intercept=float(parameters[1]),
            )
        )

        def objective(values: np.ndarray) -> float:
            return mean_squared_error(dataset.y, values[0] * dataset.x + values[1])

        numerical = finite_difference_gradient(objective, parameters)
        self.assertLess(relative_gradient_error(analytic, numerical), 1e-8)

    def test_gradient_descent_recovers_generating_line(self) -> None:
        dataset = make_linear_dataset(noise_sigma=0.05, seed=4)
        result = fit_linear_regression(dataset.x, dataset.y)
        self.assertFalse(result.diverged)
        self.assertLess(result.losses[-1], result.losses[0] * 0.01)
        self.assertAlmostEqual(result.slope, dataset.true_slope, delta=0.03)
        self.assertAlmostEqual(result.intercept, dataset.true_intercept, delta=0.03)

    def test_excessive_learning_rate_diverges(self) -> None:
        dataset = make_linear_dataset(seed=6)
        result = fit_linear_regression(dataset.x, dataset.y, learning_rate=10.0, epochs=300)
        self.assertTrue(result.diverged)


class LogisticRegressionTests(unittest.TestCase):
    def test_logistic_gradient_matches_finite_difference(self) -> None:
        dataset = make_binary_dataset(seed=3)
        parameters = np.asarray((0.1, -0.2, 0.05), dtype=np.float64)
        analytic_weights, analytic_bias = logistic_gradients(
            dataset.features,
            dataset.labels,
            weights=parameters[:2],
            bias=float(parameters[2]),
        )
        analytic = np.concatenate((analytic_weights, np.asarray((analytic_bias,))))

        def objective(values: np.ndarray) -> float:
            probabilities = sigmoid(dataset.features @ values[:2] + values[2])
            return binary_cross_entropy(dataset.labels, probabilities)

        numerical = finite_difference_gradient(objective, parameters)
        self.assertLess(relative_gradient_error(analytic, numerical), 1e-8)

    def test_logistic_training_separates_the_clusters(self) -> None:
        dataset = make_binary_dataset(seed=9)
        result = fit_logistic_regression(dataset.features, dataset.labels)
        predictions = result.predict(dataset.features)
        self.assertLess(result.losses[-1], result.losses[0] * 0.2)
        self.assertGreater(float(np.mean(predictions == dataset.labels)), 0.97)


class ExperimentTests(unittest.TestCase):
    def test_cli_experiment_writes_machine_readable_results_and_plots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            payload = run_experiment(output)
            saved = json.loads((output / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(saved, payload)
            self.assertTrue((output / "linear_loss.svg").exists())
            self.assertTrue((output / "logistic_loss.svg").exists())
            self.assertLess(saved["linear_regression"]["gradient_relative_error"], 1e-8)
            self.assertLess(saved["logistic_regression"]["gradient_relative_error"], 1e-8)


if __name__ == "__main__":
    unittest.main()

