# What is a stable loss curve
You can tell the training loss curve is **stable** if it:

- generally decreases as iterations increase
- eventually flattens out
- does not grow without bound
- does not show large, continuous jumps or oscillations

For your plot:

- A smooth decrease toward a constant value means gradient descent is converging stably.
- Large up-and-down oscillations may mean the learning rate is too high.
- An increasing loss usually means the learning rate is too high or the gradients are incorrect.
- A nearly horizontal curve from the beginning may mean the learning rate is too small or the model is barely updating.

A suitable notebook answer would be:

> The training loss curve is stable if it decreases smoothly and eventually levels off. This indicates that gradient descent is converging toward a minimum. If the curve oscillates strongly or increases, the learning rate may be too high or there may be an error in the gradient calculation.