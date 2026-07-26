"""Run the complete optimization learning experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data import make_binary_dataset, make_linear_dataset
from .gradient_check import finite_difference_gradient, relative_gradient_error
from .linear import fit_linear_regression, linear_gradients, mean_squared_error
from .logistic import binary_cross_entropy, fit_logistic_regression, logistic_gradients, sigmoid
from .svg import save_line_plot


def run_experiment(output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    linear_data = make_linear_dataset()
    linear_fit = fit_linear_regression(linear_data.x, linear_data.y)
    analytic_linear = np.asarray(
        linear_gradients(linear_data.x, linear_data.y, slope=0.4, intercept=-0.2)
    )

    def linear_objective(parameters: np.ndarray) -> float:
        predictions = parameters[0] * linear_data.x + parameters[1]
        return mean_squared_error(linear_data.y, predictions)

    numerical_linear = finite_difference_gradient(
        linear_objective,
        np.asarray((0.4, -0.2), dtype=np.float64),
    )
    excessive_fit = fit_linear_regression(
        linear_data.x,
        linear_data.y,
        learning_rate=10.0,
        epochs=300,
    )

    binary_data = make_binary_dataset()
    logistic_fit = fit_logistic_regression(binary_data.features, binary_data.labels)
    starting_weights = np.asarray((0.15, -0.2), dtype=np.float64)
    starting_bias = 0.05
    analytic_weights, analytic_bias = logistic_gradients(
        binary_data.features,
        binary_data.labels,
        weights=starting_weights,
        bias=starting_bias,
    )

    def logistic_objective(parameters: np.ndarray) -> float:
        probabilities = sigmoid(binary_data.features @ parameters[:2] + parameters[2])
        return binary_cross_entropy(binary_data.labels, probabilities)

    numerical_logistic = finite_difference_gradient(
        logistic_objective,
        np.concatenate((starting_weights, np.asarray((starting_bias,)))),
    )
    analytic_logistic = np.concatenate((analytic_weights, np.asarray((analytic_bias,))))
    predicted_labels = logistic_fit.predict(binary_data.features)
    accuracy = float(np.mean(predicted_labels == binary_data.labels))

    payload: dict[str, object] = {
        "linear_regression": {
            "true_slope": linear_data.true_slope,
            "true_intercept": linear_data.true_intercept,
            "learned_slope": linear_fit.slope,
            "learned_intercept": linear_fit.intercept,
            "initial_loss": linear_fit.losses[0],
            "final_loss": linear_fit.losses[-1],
            "gradient_relative_error": relative_gradient_error(
                analytic_linear,
                numerical_linear,
            ),
        },
        "learning_rate_experiment": {
            "learning_rate": 10.0,
            "diverged": excessive_fit.diverged,
            "recorded_steps": len(excessive_fit.losses),
        },
        "logistic_regression": {
            "weights": logistic_fit.weights.tolist(),
            "bias": logistic_fit.bias,
            "initial_loss": logistic_fit.losses[0],
            "final_loss": logistic_fit.losses[-1],
            "training_accuracy": accuracy,
            "gradient_relative_error": relative_gradient_error(
                analytic_logistic,
                numerical_logistic,
            ),
        },
    }
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    np.savetxt(output_dir / "linear_loss.csv", np.asarray(linear_fit.losses), delimiter=",")
    np.savetxt(output_dir / "logistic_loss.csv", np.asarray(logistic_fit.losses), delimiter=",")
    save_line_plot(np.asarray(linear_fit.losses), output_dir / "linear_loss.svg", title="Linear regression loss")
    save_line_plot(
        np.asarray(logistic_fit.losses),
        output_dir / "logistic_loss.svg",
        title="Logistic regression loss",
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Learn gradient descent by running inspectable experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="run linear and logistic experiments")
    run_parser.add_argument("--out", type=Path, default=Path("outputs/demo"))
    args = parser.parse_args()
    if args.command == "run":
        payload = run_experiment(args.out)
        linear = payload["linear_regression"]
        logistic = payload["logistic_regression"]
        print(
            f"Linear: slope={linear['learned_slope']:.4f} "
            f"intercept={linear['learned_intercept']:.4f} "
            f"loss={linear['final_loss']:.6f}"
        )
        print(
            f"Logistic: accuracy={logistic['training_accuracy']:.3f} "
            f"loss={logistic['final_loss']:.6f}"
        )
        print(f"Saved experiment to {args.out}")
        return 0
    raise AssertionError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

