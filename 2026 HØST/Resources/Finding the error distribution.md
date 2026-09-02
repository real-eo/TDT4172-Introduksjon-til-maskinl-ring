`y_pred` and `y` must be compared on the same set of samples, not on different sizes.

For the regression task, if:

- `X` has shape `(1000, 1)`
- `y` has shape `(1000,)`

then:
- `y_pred` should also be length `1000`

not something much larger or smaller.

So the comparison is:

```python
y_pred = model.predict(X)
errors = y - y_pred
```

and this works only if `model.predict(X)` returns one prediction per row in the same `X`.

