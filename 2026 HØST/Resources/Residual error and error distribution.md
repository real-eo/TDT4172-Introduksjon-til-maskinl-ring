A residual is the difference between the actual value and the prediction for one sample.

For regression:

$$
r_i = y_i - \hat{y}_i
$$

or equivalently:

$$
r_i = \text{true value} - \text{predicted value}
$$

This is the error for one sample.

---

## Example

If:
- actual energy = 80
- predicted energy = 75

then the residual is:

$$
80 - 75 = 5
$$

This means the model slightly underpredicted for that point.

If the residual is negative:
- actual < predicted
- model overpredicted

If the residual is positive:
- actual > predicted
- model underpredicted

---

## Why residuals matter

Residuals let you inspect the error distribution:
- centered around zero → good
- large spread → poor
- skewed → systematic bias

This is exactly what the assignment is asking for in the “error distribution” question.

---

## In your notebook

This is the main code:

```python
y_pred = linearRegressionModel.predict(X)
errors = y - y_pred
```

Then `errors` is a vector of residuals, one per sample.

So:
- `errors` = residuals
- `plt.hist(errors)` = histogram of residuals
- `errors.mean()` = average residual
- `errors.std()` = spread of residuals

---

## Short version

Residual = one sample’s prediction error.

That is the concept behind the error distribution plot.