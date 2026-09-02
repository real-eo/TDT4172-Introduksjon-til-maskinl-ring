# Why logistic regression models used `predict()` and `predict_proba()`
We use `predict_proba()` because logistic regression’s probability output must be between $0$ and $1$.

The raw linear model computes:
$$
z = Xw+b
$$

But $z$ can be any real number. The sigmoid transforms it into a probability:

$$
p = \sigma(z)=\frac{1}{1+e^{-z}}
$$

That is why `predict_proba` does:

```python
z = np.dot(transformedX, self.weights) + self.bias
return self.sigmoid(z)
```

Then `predict` converts the probability into a class:

```python
return (probabilities >= 0.5).astype(int)
```

For example:

```text
probability = 0.82  -> class 1
probability = 0.31  -> class 0
```

The threshold `0.5` is used because it treats both classes equally. `predict_proba` is kept separate because probabilities are needed for ROC curves and AUC, while `predict` returns only the final labels.