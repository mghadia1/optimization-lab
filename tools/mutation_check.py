#!/usr/bin/env python3
"""Fault injection: does this test suite actually catch bugs?

"6 tests pass" says nothing on its own. A suite of six assertions that never
fail also passes. The claim worth making is *"6 tests, and every one of N
deliberately injected faults is caught"* — which requires injecting the faults
and counting.

Each mutation below is a plausible mistake in this project's arithmetic: a
dropped factor of two, a sign flip, a copy-paste between two gradients, a
divergence guard that never trips. The harness copies the project to a scratch
directory, applies one mutation, runs the suite there, and records whether the
suite failed. A mutation the suite still passes is a hole in the tests, and is
reported as such rather than quietly dropped.

Two guards make the count mean something:

  * a control run with no mutation must PASS. If the suite fails for unrelated
    reasons, every mutation would look "caught" and the score would be a lie.
  * each mutation's target text must appear EXACTLY once in the file. If the
    source moves on and a pattern stops matching, the harness errors instead of
    silently skipping the fault and reporting a smaller, flattering total.

Bytecode is disabled in the child runs. A stale .pyc can keep executing code
that is no longer on disk, which makes a suite report on a file it is not
actually testing.

    python tools/mutation_check.py            # human-readable table
    python tools/mutation_check.py --json     # machine-readable, for the README
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Mutation:
    identifier: str
    module: str
    original: str
    mutated: str
    fault: str


MUTATIONS: tuple[Mutation, ...] = (
    Mutation(
        "linear-slope-gradient-scale",
        "linear.py",
        "slope_gradient = 2.0 * float(np.mean(residuals * x_values))",
        "slope_gradient = 1.0 * float(np.mean(residuals * x_values))",
        "drops the factor of 2 from d(MSE)/d(slope); descent still converges, just half as fast",
    ),
    Mutation(
        "linear-intercept-gradient-scale",
        "linear.py",
        "intercept_gradient = 2.0 * float(np.mean(residuals))",
        "intercept_gradient = 1.0 * float(np.mean(residuals))",
        "same dropped factor on the intercept gradient",
    ),
    Mutation(
        "linear-intercept-gradient-copy-paste",
        "linear.py",
        "intercept_gradient = 2.0 * float(np.mean(residuals))",
        "intercept_gradient = 2.0 * float(np.mean(residuals * x_values))",
        "intercept gradient copy-pasted from the slope gradient",
    ),
    Mutation(
        "linear-residual-sign",
        "linear.py",
        "residuals = slope * x_values + intercept - y_values",
        "residuals = y_values - slope * x_values - intercept",
        "residual sign flipped, so gradient descent walks uphill",
    ),
    Mutation(
        "linear-descent-becomes-ascent",
        "linear.py",
        "slope -= learning_rate * slope_gradient",
        "slope += learning_rate * slope_gradient",
        "update adds the gradient instead of subtracting it",
    ),
    Mutation(
        "linear-mse-becomes-sse",
        "linear.py",
        "return float(np.mean(residuals * residuals))",
        "return float(np.sum(residuals * residuals))",
        "mean squared error silently becomes a sum, so the loss scales with n",
    ),
    Mutation(
        "linear-divergence-never-flagged",
        "linear.py",
        "            if not np.isfinite(loss):\n                diverged = True",
        "            if not np.isfinite(loss):\n                diverged = False",
        "a diverged run reports itself as converged",
    ),
    Mutation(
        "logistic-gradient-not-averaged",
        "logistic.py",
        "weight_gradient = feature_values.T @ errors / len(label_values)",
        "weight_gradient = feature_values.T @ errors",
        "weight gradient summed instead of averaged, so it scales with batch size",
    ),
    Mutation(
        "logistic-descent-becomes-ascent",
        "logistic.py",
        "weights -= learning_rate * weight_gradient",
        "weights += learning_rate * weight_gradient",
        "logistic update maximises the loss instead of minimising it",
    ),
    Mutation(
        "finite-difference-halved",
        "gradient_check.py",
        "result[index] = (objective(positive) - objective(negative)) / (2.0 * epsilon)",
        "result[index] = (objective(positive) - objective(negative)) / epsilon",
        "the gradient checker itself breaks: a centered difference divided by one step",
    ),
)


def _copy_project(destination: Path) -> None:
    """Copy just enough of the project to run its suite."""
    for item in ("src", "tests", "pyproject.toml"):
        source = PROJECT_ROOT / item
        target = destination / item
        if source.is_dir():
            shutil.copytree(
                source, target, ignore=shutil.ignore_patterns("__pycache__", "*.egg-info", "*.pyc")
            )
        else:
            shutil.copy2(source, target)


def _run_suite(project: Path) -> tuple[bool, str]:
    """Run the suite inside `project`. Returns (passed, last line of output)."""
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    # The copied tests/ directory contains the test that invokes this harness.
    # Without both of these the child would run it, spawning its own children,
    # forever. The ignore handles it; the env var is the backstop if the file is
    # ever renamed.
    environment["OPTIMIZATION_LAB_MUTATION_CHILD"] = "1"
    completed = subprocess.run(
        [
            sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider",
            "--ignore=tests/test_mutation_coverage.py",
        ],
        cwd=project,
        capture_output=True,
        text=True,
        env=environment,
    )
    output = (completed.stdout + completed.stderr).strip().splitlines()
    summary = output[-1] if output else "(no output)"
    return completed.returncode == 0, summary


def _apply(project: Path, mutation: Mutation) -> None:
    path = project / "src" / "optimization_lab" / mutation.module
    text = path.read_text()
    occurrences = text.count(mutation.original)
    if occurrences != 1:
        raise SystemExit(
            f"mutation {mutation.identifier!r}: its target appears {occurrences} times in "
            f"{mutation.module} (expected exactly 1). The source moved; update the mutation "
            f"rather than letting the harness skip a fault it can no longer inject."
        )
    path.write_text(text.replace(mutation.original, mutation.mutated))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit a machine-readable report")
    arguments = parser.parse_args()

    results = []
    with tempfile.TemporaryDirectory(prefix="optimization-lab-mutation-") as scratch:
        control = Path(scratch) / "control"
        control.mkdir()
        _copy_project(control)
        control_passed, control_summary = _run_suite(control)
        if not control_passed:
            raise SystemExit(
                "control run FAILED before any mutation was applied "
                f"({control_summary}). Every mutation would look 'caught'. Fix the suite first."
            )

        for index, mutation in enumerate(MUTATIONS):
            workspace = Path(scratch) / f"mutant-{index:02d}"
            workspace.mkdir()
            _copy_project(workspace)
            _apply(workspace, mutation)
            passed, summary = _run_suite(workspace)
            results.append(
                {
                    "id": mutation.identifier,
                    "module": mutation.module,
                    "fault": mutation.fault,
                    "caught": not passed,
                    "suite": summary,
                }
            )

    caught = sum(1 for row in results if row["caught"])
    report = {
        "control": control_summary,
        "mutations": len(results),
        "caught": caught,
        "results": results,
    }

    if arguments.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"control (no mutation): {control_summary}\n")
        for row in results:
            mark = "caught " if row["caught"] else "MISSED "
            print(f"  {mark} {row['id']:<38} {row['suite']}")
        print(f"\n{caught}/{len(results)} injected faults caught")
        if caught != len(results):
            print("\nA missed fault is a hole in the suite, not a rounding error.")
            for row in results:
                if not row["caught"]:
                    print(f"  - {row['id']}: {row['fault']}")

    return 0 if caught == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
