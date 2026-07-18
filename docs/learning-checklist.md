# Learning checklist

Do not memorize definitions. Answer these questions while pointing to the code.

1. What is a model parameter in linear regression?
2. Why is the residual defined as prediction minus target?
3. Why does gradient descent subtract the gradient?
4. What happens when the learning rate is too large?
5. Why can a finite-difference check find an incorrect derivative?
6. What does the sigmoid change about a linear score?
7. Why is binary cross-entropy useful for probabilities?
8. Why is the logistic training accuracy not a real test-set result?
9. What is the difference between an epoch and a learning rate?
10. Which line performs each parameter update?

## Required personal change

Run the baseline, then change the linear learning rate to `0.01`. Record:

- the initial loss;
- the loss after 20 steps;
- the final loss;
- why the final result may be correct even though early progress is slower.

Next, try a value above `1.0` and explain the loss behavior without describing divergence as a software crash.

