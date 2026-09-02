## BGD
**BGD** means **Batch Gradient Descent**.

It is an optimization method where the model uses **the entire training dataset** to calculate the gradient before each parameter update.

For each iteration:

1. Make predictions for all samples.
2. Calculate the loss for all samples.
3. Calculate the average gradient.
4. Update the weights and bias.

$$
\theta_{\text{new}}
=
\theta_{\text{old}}
-
\eta \nabla J(\theta)
$$

where $\eta$ is the learning rate.

**Difference from SGD:**

- **BGD:** uses all training samples per update; stable but potentially slower.
- **SGD:** uses one sample per update; faster updates but noisier.
- **Mini-batch gradient descent:** uses a small group of samples per update.

If your `computeGradients` function uses all of `X` and `y` at once, your implementation is using **BGD**, even if the assignment refers to SGD generally.