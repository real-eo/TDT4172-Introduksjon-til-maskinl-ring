# Cost vs. Loss
They are closely related, but technically:

- **Loss**: the error for one individual sample.
- **Cost**: the average loss over the entire training dataset.

For mean squared error:

$$
\text{loss}_i = (y_i - \hat{y}_i)^2
$$

$$
\text{cost} = \frac{1}{m}\sum_{i=1}^{m}(y_i-\hat{y}_i)^2
$$

In practice, people often use “loss” and “cost” interchangeably. In your code, `calculateCost` computes the average squared error across all samples, so it is technically the **cost**, or dataset-level loss.