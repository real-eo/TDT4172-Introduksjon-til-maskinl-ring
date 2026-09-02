import numpy as np

class LinearRegression():
    # * Strip away the buzzwords first
    # ŷ = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
    #   - x₁...xₙ -- our features (network activity metrics: packets/sec, bytes sent, connections open, etc.)
    #   - w₁...wₙ -- the weights: how much each feature matters. This is what we're solving for.
    #   - b -- the bias (intercept): the baseline output when all features are zero.
    #   - ŷ (y-hat) -- our prediction (estimated power consumption)
    #   - y -- the actual value from our dataset (ground truth)
    # 
    # That's the entire model. "Training" means: find the ws and b that make ŷ close to y across all our data rows.
    
    # * How "close" is measured -- the cost function
    # We need a single number that tells us how wrong the model currently 
    # is. The standard one for linear regression is Mean Squared Error (MSE):
    # J(w, b) = (1/m) * Σ(ŷᵢ - yᵢ)²
    #   - m = number of training examples (rows in our dataset)
    #   - You square the error so positive and negative errors don't cancel out, and so bigger errors are punished more
    #   - J is just the conventional letter for "cost function" -- no deeper meaning
    # 
    # This J is a function of our weights and bias. Training = finding the w, b that minimize J.
    
    # * How we minimize it -- gradient descent
    # We can't just guess. We compute the gradient (the vector of partial derivatives of J with respect to each weight 
    # and the bias) -- this tells us the direction of steepest increase in error. You step in the opposite direction.
    # The derivatives:
    #   - ∂J/∂wⱼ = (2/m) * Σ(ŷᵢ - yᵢ) * xᵢⱼ
    #   - ∂J/∂b  = (2/m) * Σ(ŷᵢ - yᵢ)
    #
    # Update rule, applied every iteration:
    #       wⱼ := wⱼ - α * ∂J/∂wⱼ
    #       b  := b  - α * ∂J/∂b
    #   - α (alpha) = learning rate -- how big a step we take. Too big → diverges/oscillates. Too small → painfully slow.
    #   - One full pass of computing gradients over all m examples and updating = one epoch (or "iteration," people use these loosely)
    #   - Convergence = when J stops decreasing meaningfully between epoch
    # 
    # That's it. Everything else in ML (neural nets, etc.) is this same loop with a fancier ŷ.
    
    # * Feature scaling
    # The features will be wildly different scales -- "bytes sent" might be in the millions, "open connections" might be 0–50. Gradient 
    # descent converges badly (or not at all) with unscaled features because the cost surface becomes a narrow ravine. So we normalize:
    #       x_scaled = (x - mean) / std_dev      // standardization
    #
    # We do this per-feature, store the mean/std, and remember to apply the same transform to any new data we predict on later (and 
    # un-scale our final predictions/weights if we want them in original units).
    
    def __init__(self, lr=0.001, n_iterations=1000, scale_features=True):
        self.weights: np.ndarray = None
        self.bias: float = None
        
        self.lr: float = lr
        self.n_iterations: int = n_iterations
        
        self.loss_history: list[float] = []
        
        # ? Added attributes for feature scaling
        self.scale_features: bool = scale_features
        self.mean: np.ndarray = None
        self.standardDeviation: np.ndarray = None
        
        if self.scale_features:
            print("WARNING: Feature scaling is enabled! This can result in needing drastically higher/lower learning rate- and iteration count compared to the defaults.")
  
    @property
    def unscaledWeights(self) -> np.ndarray:
        if not self.scale_features: return self.weights
        return self.weights / self.standardDeviation

    @property
    def unscaledBias(self) -> float:
        if not self.scale_features: return self.bias
        return (
            self.bias 
            - np.sum(
                (self.weights * self.mean) 
                / self.standardDeviation
            )
        )
        
    def calculateCost(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # * J(w, b) = (1/m) * Σ(ŷᵢ - yᵢ)²
        #   - m = number of training examples (rows in our dataset)
        #   - You square the error so positive and negative errors don't cancel out, and so bigger errors are punished more
        #   - J is just the conventional letter for "cost function" -- no deeper meaning

        m = y_true.shape[0]
        _cost = (1/m) * np.sum((y_pred - y_true) ** 2)
        
        return _cost
    
    def computeGradients(self, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, float]:
        # * ∂J/∂wⱼ = (2/m) * Σ(ŷᵢ - yᵢ) * xᵢⱼ
        # * ∂J/∂b  = (2/m) * Σ(ŷᵢ - yᵢ)
        m = y_true.shape[0]
        dw = (2/m) * np.dot(X.T, (y_pred - y_true))
        db = (2/m) * np.sum(y_pred - y_true)
        
        return dw, db
    
    def scaleFeatures(self, X: np.ndarray, training: bool = False) -> np.ndarray:
        # * x_scaled = (x - mean) / std_dev                                             // standardization
        
        # // print("X:", X.shape)
        # // print(X[:5])
        
        # Update mean and standard deviation only during training
        if training:
            self.mean = np.mean(X, axis=0)
            self.standardDeviation = np.std(X, axis=0)
        
        # Scale the features using the stored mean and standard deviation
        xScaled = (X - self.mean) / self.standardDeviation
        
        # // print("xScaled:", xScaled.shape)
        # // print(xScaled[:5]) 
        
        return xScaled


    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats
        """
        # 1. Scale the features of X using self.scaleFeatures() if self.scale_features is True
        # 2. Initialize self.weights and self.bias to zeros
        # 3. For each iteration:
        #    a. Compute predictions using the current weights and bias
        #    b. Compute the cost using self.calculateCost()
        #    c. Compute gradients using self.computeGradients()
        #    d. Update weights and bias using the gradients and learning rate
        
        # * 1. Scale the features of X using self.scaleFeatures() 
        if self.scale_features: transformedX = self.scaleFeatures(X, training=True)     # Update mean and standard deviation
        else:                   transformedX = X

        # * 2. Initialize self.weights and self.bias to zeros
        self.weights = np.zeros(transformedX.shape[1])                                  # NOTE: This is a 1D array of length n in the assignment
        self.bias = 0
        
        # * 3. Iterate for the specified number of iterations
        print(f"Training started for {self.n_iterations} iterations...")
        
        # ŷ = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
        #   - x₁...xₙ -- our features (network activity metrics: packets/sec, bytes sent, connections open, etc.)
        #   - w₁...wₙ -- the weights: how much each feature matters. This is what we're solving for.
        #   - b -- the bias (intercept): the baseline output when all features are zero.
        #   - ŷ (y-hat) -- our prediction (estimated power consumption)
        #   - y -- the actual value from our dataset (ground truth)
        for i in range(self.n_iterations):    
            # a. Compute predictions using the current weights and bias
            yPred = np.dot(transformedX, self.weights) + self.bias

            # b. Compute the cost using self.calculateCost()
            cost = self.calculateCost(y, yPred)
            self.loss_history.append(cost)

            # c. Compute gradients using self.computeGradients()
            dw, db = self.computeGradients(transformedX, y, yPred)
            
            # d. Update weights and bias using the gradients and learning rate
            #       wⱼ := wⱼ - α * ∂J/∂wⱼ
            #       b  := b  - α * ∂J/∂b
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
        # | Print the final cost after training
        print(f"Training completed. Final cost: {self.loss_history[-1]}")
        
    
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats
        """
        # 1. Scale the features of X using the mean and std from training
        # 2. Compute predictions using the current weights and bias
        
        # * 1. Scale the features of X using the mean and std from training
        if self.scale_features: transformedX = self.scaleFeatures(X)
        else:                   transformedX = X
        
        # // print("Scaled X for prediction:", scaledX.shape)
        
        # * 2. Compute predictions using the current weights and bias
        yPred = np.dot(transformedX, self.weights) + self.bias

        return yPred


class LogisticRegression(): 
    # * Logistic regression  
    # For logistic regression, the model is similar to linear regression, but we apply a 
    # sigmoid function to the linear combination of features to produce a probability output. 
    #
    # We redefine the linear ŷ to z, as ŷ represent the final model prediction. Then we
    # apply the sigmoid function to z to get the probability. This probability is the new
    # ŷ, which is the output of the logistic regression model.
    #   z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b      ← identical to before, just renamed z
    #   ŷ = 1 / (1 + e^(-z))                 ← sigmoid squashes z into (0, 1)
    
    # * The cost function
    # For logistic regression, we use the binary cross-entropy loss function, which is defined as:
    # J(w, b) = -(1/m) * Σ[yᵢ * log(ŷᵢ) + (1 - yᵢ) * log(1 - ŷᵢ)]
    #   - m = number of training examples
    #   - yᵢ = actual label (0 or 1)
    #   - ŷᵢ = predicted probability for the positive class (output of the sigmoid function)
    #  
    # This loss function penalizes incorrect predictions more heavily, especially when the model is confident but wrong.

    # * Gradients
    # Finding the gradients for logistic regression is the same as for linear regression, we take the
    # partial derivatives of the cost function with respect to each weight and the bias. The gradients are:
    #   - ∂J/∂wⱼ = (1/m) * Σ(ŷᵢ - yᵢ) * xᵢⱼ
    #   - ∂J/∂b  = (1/m) * Σ(ŷᵢ - yᵢ)
    #
    # The update rules are therefore the same as in linear regression:
    #       wⱼ := wⱼ - α * ∂J/∂wⱼ
    #       b  := b  - α * ∂J/∂b
    
    # * Feature scaling
    # Since feature scaling is only dependent on the input features, it is the same as in a linear regression model.
      
    def __init__(self, lr=0.001, n_iterations=1000):
        self.weights = None
        self.bias = None
        
        self.lr = lr
        self.n_iterations = n_iterations
        
        self.loss_history = []
    
    def fit(self, X, y):
        # ====================================
        # YOUR CODE GOES HERE
        # ====================================
        raise NotImplementedError("LogisticRegression.fit is not implemented yet.")
    
    def predict_proba(self, X):
        # ====================================
        
        # YOUR CODE GOES HERE
        # ====================================
        raise NotImplementedError("LogisticRegression.predict_proba is not implemented yet.")
        
    def predict(self, X):
        # ====================================
        # YOUR CODE GOES HERE
        # ====================================
        raise NotImplementedError("LogisticRegression.predict is not implemented yet.")
    
    def sigmoid(self, z):
        # ====================================
        # YOUR CODE GOES HERE
        # ====================================
        raise NotImplementedError("LogisticRegression.sigmoid is not implemented yet.")
        