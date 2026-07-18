# Optimization Lab

Optimization Lab implements linear and logistic regression with NumPy so the training process is visible.

> **What this is not:** a headline ML project. It is a learning artifact for understanding gradients before using PyTorch or scikit-learn. It contains no medical data and makes no medical claim.

![Linear regression training-loss curve from the recorded demo run](docs/hero.svg)

## Quickstart

```bash
git clone https://github.com/mghadia1/optimization-lab.git
cd optimization-lab
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -e .

# run the tests (6)
PYTHONPATH=src python -m unittest discover -s tests -v

# run the demo (writes results.json, loss CSVs, and SVG loss plots)
PYTHONPATH=src python -m optimization_lab.cli run --out outputs/demo
```

## Headline result

All six automated tests pass. They verify the hand-derived analytic gradients against centered finite differences, confirm the loss falls on suitable synthetic data, and confirm that a deliberately excessive learning rate diverges.

## What it demonstrates

- deterministic synthetic regression and binary-classification datasets;
- mean squared error and binary cross-entropy;
- hand-derived batch gradients;
- centered finite-difference gradient checks;
- gradient descent for linear regression;
- logistic regression with a numerically stable sigmoid;
- a deliberately excessive learning-rate experiment;
- automated tests and machine-readable experiment output.

## Learning order

1. Read `docs/how-it-works.md` beside the source.
2. Run the tests and demo.
3. Change one learning rate and predict what will happen before running it.
4. Complete the questions in `docs/learning-checklist.md` in your own words.

This repository should normally be linked from a portfolio rather than used as a headline resume project. It contains no medical data and makes no medical claim.

