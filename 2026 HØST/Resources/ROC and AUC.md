# ROC and AUC

Think of the model as giving every observation a **score** between 0 and 1:

```python
probabilities = logisticRegressionModel.predict_proba(xTest)
```

For example:

| Actual class | Predicted probability |
|---|---:|
| 1 | 0.90 |
| 1 | 0.75 |
| 0 | 0.30 |
| 0 | 0.10 |

The model ranks the positive examples above the negative examples, so it separates the classes well.

The threshold decides where to draw the line:

- Threshold `0.5`: probabilities above `0.5` become class `1`
- Threshold `0.8`: only probabilities above `0.8` become class `1`
- Threshold `0.2`: more observations become class `1`

For every possible threshold, we calculate:

- **True positive rate**: how many actual positives were found
- **False positive rate**: how many actual negatives were incorrectly selected

The ROC curve plots these values. A model with good separation rises quickly toward the top-left corner. A random model follows the diagonal line.

AUC summarizes the entire ROC curve as one number. Another intuitive interpretation is:

> AUC is the probability that the model gives a randomly chosen positive example a higher score than a randomly chosen negative example.

So:

- AUC `1.0`: all positives receive higher scores than all negatives
- AUC `0.5`: the scores are essentially random
- AUC below `0.5`: the model tends to rank the classes backwards

Accuracy is different because it uses only one threshold, usually `0.5`:

```python
predictions = probabilities >= 0.5
```

AUC evaluates the quality of the probability ranking across **all thresholds**, which is why it does not depend only on `0.5`.