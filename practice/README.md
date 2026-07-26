# Optimization Lab — practice (code it yourself)

Reimplement the math cores from scratch. The working project in `../src/` stays
untouched and green; this folder is your sandbox.

## What to implement (in this order)

Fill in the `TODO`s. The plumbing (dataclasses, `_validated_*` helpers, `cli.py`,
`svg.py`, `data.py`) is done — leave it alone and focus on the concepts.

1. `my_optimization_lab/linear.py` → `mean_squared_error`  (TODO 1)
2. `my_optimization_lab/linear.py` → `linear_gradients`     (TODO 2)
3. `my_optimization_lab/gradient_check.py` → both functions (TODO 3a, 3b)
   — after this, the gradient-check test proves your TODO 2 calculus is right.
4. `my_optimization_lab/linear.py` → `fit_linear_regression` (TODO 4)
5. `my_optimization_lab/logistic.py` → `sigmoid`             (TODO 5)
6. `my_optimization_lab/logistic.py` → `binary_cross_entropy`(TODO 6)
7. `my_optimization_lab/logistic.py` → `logistic_gradients`  (TODO 7)
8. `my_optimization_lab/logistic.py` → `fit_logistic_regression` (TODO 8)

## Grade yourself

From this `practice/` folder, with the project venv active:

```bash
cd "/Users/programming/Documents/auto job applier/projects/optimization-lab"
source .venv/bin/activate
cd practice
PYTHONPATH=. python -m unittest test_practice -v
```

Run it after each function. Which test maps to what:

| Test | Needs |
|---|---|
| `test_analytic_gradient_matches_finite_difference` | TODO 1, 2, 3a, 3b |
| `test_gradient_descent_recovers_generating_line` | TODO 4 |
| `test_excessive_learning_rate_diverges` | TODO 4 (the divergence guard) |
| `test_logistic_gradient_matches_finite_difference` | TODO 5, 6, 7, 3a, 3b |
| `test_logistic_training_separates_the_clusters` | TODO 5..8 |
| `test_cli_experiment_writes_...` | everything (runs the full demo) |

## Stuck?

The committed reference is your answer key:

```bash
git show HEAD:src/optimization_lab/linear.py
git show HEAD:src/optimization_lab/logistic.py
git show HEAD:src/optimization_lab/gradient_check.py
```

Paste me your version and I'll tell you *why* it's off — I won't just hand you the fix.
