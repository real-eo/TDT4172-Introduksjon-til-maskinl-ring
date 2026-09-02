## SGD
**SGD** means **Stochastic Gradient Descent**. It is an optimization method used to find the model parameters that minimize the loss.

For each training step, it:

1. Calculates predictions.
2. Calculates the gradient, which shows how the weights and bias should change.
3. Updates the parameters in the opposite direction of the gradient.

For a parameter $\theta$:

$$
\theta_{\text{new}} = \theta_{\text{old}} - \text{learning rate} \times \text{gradient}
$$

“Stochastic” means the model updates its parameters using one sample or a small batch at a time, instead of the entire dataset.

In your implementation, if `fit` uses all training samples to calculate each gradient, it is technically **batch gradient descent**, not SGD. The basic idea is the same: repeatedly adjust the weights and bias to reduce the cost.



## When the gradient in SGD is close to zero
When the **gradient in SGD is close to zero**, it means the model parameters are near a point where changing them slightly will not reduce the loss much.

In other words:

- the weights and bias are close to an optimum
- the loss curve is flattening
- the model is making progress slowly
- the current predictions are reasonably suited to the training data

For your linear regression:

$$
w_{\text{new}} = w - \eta \frac{\partial J}{\partial w}
$$

If $\frac{\partial J}{\partial w} \approx 0$, the update to $w$ is very small.

However, a gradient close to zero does **not always** guarantee a good model. It can also happen if:
- the learning rate is too small
- the model is stuck at a local or flat point
- the data was scaled incorrectly
- the model has converged to a poor solution

For your simple linear regression with mean squared error, it usually indicates that gradient descent has converged toward the best-fitting line.