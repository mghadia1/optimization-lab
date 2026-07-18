# How Optimization Lab works

## The idea behind training

A model contains parameters. A loss function measures how wrong its predictions are. A gradient tells us how the loss changes when a parameter changes. Gradient descent moves each parameter in the opposite direction of its gradient:

```text
new parameter = old parameter - learning rate * gradient
```

The learning rate controls the step size. A tiny value learns slowly. A suitable value decreases the loss. An excessive value can jump across the minimum and diverge.

## Linear regression

The model predicts:

```text
predicted y = slope * x + intercept
```

The residual for one sample is `predicted y - actual y`. Mean squared error averages the squared residuals. Squaring makes negative and positive errors contribute positively and penalizes larger errors more strongly.

For `n` samples, the implemented gradients are:

```text
dLoss/dSlope     = (2/n) * sum(residual * x)
dLoss/dIntercept = (2/n) * sum(residual)
```

`linear_gradients` calculates those values. `fit_linear_regression` repeatedly updates the slope and intercept and records the loss before every update.

## Why gradient checking matters

The finite-difference checker slightly increases and decreases one parameter:

```text
numerical gradient = (loss(parameter + epsilon) - loss(parameter - epsilon)) / (2 * epsilon)
```

This is slower than the analytic formula, but it is independent evidence that the hand-derived formula is correct. The tests compare both gradients with a scale-aware relative error.

## Logistic regression

Logistic regression first calculates a linear score:

```text
score = features @ weights + bias
```

The sigmoid maps that score to a number between zero and one. We interpret it as the model's estimated probability for class 1. Binary cross-entropy penalizes confident wrong predictions more strongly than uncertain predictions.

The gradient has the same basic idea as linear regression: prediction error is multiplied by the input features, averaged, and used to update the weights.

## What the experiment proves

The experiment is designed to prove specific behavior, not to produce an impressive benchmark:

- the analytic and numerical gradients agree;
- a suitable learning rate recovers the known line;
- an excessive learning rate diverges;
- logistic loss decreases on two generated clusters;
- the saved JSON can be inspected by another program.

The logistic accuracy is training accuracy on an intentionally easy synthetic dataset. It is not evidence of generalization to new real data. Train/validation/test splitting is introduced in SensorGuard ML.

## Files to read

- `data.py`: creates data with a known generating process;
- `linear.py`: mean squared error, gradients, and training;
- `logistic.py`: sigmoid, cross-entropy, gradients, and training;
- `gradient_check.py`: numerical derivative checker;
- `cli.py`: runs the experiments and saves evidence;
- `tests/test_optimization.py`: protects the expected behavior.

