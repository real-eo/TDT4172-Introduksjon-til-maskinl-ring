An **unbiased estimator** is an estimator that is correct on average over many possible datasets.

For a parameter $\theta$, an estimator $\hat{\theta}$ is unbiased if:

$$
E[\hat{\theta}] = \theta
$$

In your regression, this means the model does not consistently overpredict or underpredict. The expected residual is zero:

$$
E[y - \hat{y}] = 0
$$

Your residual histogram should therefore be approximately centered around zero. A residual average near zero suggests little systematic bias, but it does not guarantee that every prediction is accurate.