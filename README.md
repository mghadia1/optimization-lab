# Optimization Lab

Optimization Lab implements linear and logistic regression with NumPy so the training process is visible. It is a learning artifact for understanding gradients before using PyTorch or scikit-learn.

## What it demonstrates

- deterministic synthetic regression and binary-classification datasets;
- mean squared error and binary cross-entropy;
- hand-derived batch gradients;
- centered finite-difference gradient checks;
- gradient descent for linear regression;
- logistic regression with a numerically stable sigmoid;
- a deliberately excessive learning-rate experiment;
- automated tests and machine-readable experiment output.

## Run it

```bash
cd projects/optimization-lab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
PYTHONPATH=src python -m optimization_lab.cli run --out outputs/demo
```

The command writes `results.json`, loss histories as CSV files, and two SVG loss plots.

## Run the tests

```bash
cd projects/optimization-lab
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Learning order

1. Read `docs/how-it-works.md` beside the source.
2. Run the tests and demo.
3. Change one learning rate and predict what will happen before running it.
4. Complete the questions in `docs/learning-checklist.md` in your own words.

This repository should normally be linked from a portfolio rather than used as a headline resume project. It contains no medical data and makes no medical claim.

