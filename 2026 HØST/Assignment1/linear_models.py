import numpy as np

class LinearRegression():
    # * Strip away the buzzwords first
    # ŷ = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
    #   - x₁...xₙ — your features (network activity metrics: packets/sec, bytes sent, connections open, etc.)
    #   - w₁...wₙ — the weights: how much each feature matters. This is what you're solving for.
    #   - b — the bias (intercept): the baseline output when all features are zero.
    #   - ŷ (y-hat) — your prediction (estimated power consumption)
    #   - y — the actual value from your dataset (ground truth)
    # 
    # That's the entire model. "Training" means: find the ws and b that make ŷ close to y across all your data rows.
    
    # * How "close" is measured — the cost function
    # We need a single number that tells you how wrong the model currently 
    # is. The standard one for linear regression is Mean Squared Error (MSE):
    # J(w, b) = (1/m) * Σ(ŷᵢ - yᵢ)²
    #   - m = number of training examples (rows in your dataset)
    #   - You square the error so positive and negative errors don't cancel out, and so bigger errors are punished more
    #   - J is just the conventional letter for "cost function" — no deeper meaning
    # 
    # This J is a function of your weights and bias. Training = finding the w, b that minimize J.
    
    # * How you minimize it — gradient descent
    # You can't just guess. You compute the gradient (the vector of partial derivatives of J with respect to each weight 
    # and the bias) — this tells you the direction of steepest increase in error. You step in the opposite direction.
    # The derivatives:
    #   - ∂J/∂wⱼ = (2/m) * Σ(ŷᵢ - yᵢ) * xᵢⱼ
    #   - ∂J/∂b  = (2/m) * Σ(ŷᵢ - yᵢ)
    #
    # Update rule, applied every iteration:
    #       wⱼ := wⱼ - α * ∂J/∂wⱼ
    #       b  := b  - α * ∂J/∂b
    #   - α (alpha) = learning rate — how big a step you take. Too big → diverges/oscillates. Too small → painfully slow.
    #   - One full pass of computing gradients over all m examples and updating = one epoch (or "iteration," people use these loosely)
    #   - Convergence = when J stops decreasing meaningfully between epoch
    # 
    # That's it. Everything else in ML (neural nets, etc.) is this same loop with a fancier ŷ.
    
    # * One more thing you'll need: feature scaling
    # Your features will be wildly different scales — "bytes sent" might be in the millions, "open connections" might be 0–50. Gradient 
    # descent converges badly (or not at all) with unscaled features because the cost surface becomes a narrow ravine. So you normalize:
    #       x_scaled = (x - mean) / std_dev      // standardization
    #
    # You do this per-feature, store the mean/std, and remember to apply the same transform to any new data you predict on later (and 
    # un-scale your final predictions/weights if you want them in original units).
    
    def __init__(self, lr=0.001, n_iterations=1000):
        self.weights: np.ndarray = None
        self.bias: float = None
        
        self.lr: float = lr
        self.n_iterations: int = n_iterations
        
        self.loss_history: list[float] = []

        
    def calculateCost(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # * J(w, b) = (1/m) * Σ(ŷᵢ - yᵢ)²
        #   - m = number of training examples (rows in your dataset)
        #   - You square the error so positive and negative errors don't cancel out, and so bigger errors are punished more
        #   - J is just the conventional letter for "cost function" — no deeper meaning

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
    
    def scaleFeatures(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # * x_scaled = (x - mean) / std_dev                                             // standardization
        
        # // print("X:", X.shape)
        # // print(X[:5])
        
        mean = np.mean(X, axis=0)
        standardDeviation = np.std(X, axis=0)
        xScaled = (X - mean) / standardDeviation
        
        # // print("xScaled:", xScaled.shape)
        # // print(xScaled[:5]) 
        
        return xScaled, mean, standardDeviation
    

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats
        """
        # 1. Scale the features of X using self.scaleFeatures()
        # 2. Initialize self.weights and self.bias to zeros
        # 3. For each iteration:
        #    a. Compute predictions using the current weights and bias
        #    b. Compute the cost using self.calculateCost()
        #    c. Compute gradients using self.computeGradients()
        #    d. Update weights and bias using the gradients and learning rate
        
        # * 1. Scale the features of X using self.scaleFeatures() 
        scaledX, mean, std = self.scaleFeatures(X)
        # // print("Scaled X:", scaledX.shape)
        # // print("y:", y.shape)

        # * 2. Initialize self.weights and self.bias to zeros
        self.weights = np.zeros(scaledX.shape[1])                                       # NOTE: This is a 1D array of length n in the assignment
        self.bias = 0
        
        # * 3. Iterate for the specified number of iterations
        # ŷ = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
        #   - x₁...xₙ — your features (network activity metrics: packets/sec, bytes sent, connections open, etc.)
        #   - w₁...wₙ — the weights: how much each feature matters. This is what you're solving for.
        #   - b — the bias (intercept): the baseline output when all features are zero.
        #   - ŷ (y-hat) — your prediction (estimated power consumption)
        #   - y — the actual value from your dataset (ground truth)
        for i in range(self.n_iterations):    
            # a. Compute predictions using the current weights and bias
            yPred = np.dot(scaledX, self.weights) + self.bias

            # b. Compute the cost using self.calculateCost()
            cost = self.calculateCost(y, yPred)
            self.loss_history.append(cost)

            # c. Compute gradients using self.computeGradients()
            dw, db = self.computeGradients(scaledX, y, yPred)
            
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
        scaledX, _, _ = self.scaleFeatures(X)                                           # ? Only take the scaled features, ignore mean and std
        
        # // print("Scaled X for prediction:", scaledX.shape)
        
        # * 2. Compute predictions using the current weights and bias
        yPred = np.dot(scaledX, self.weights) + self.bias

        return yPred


class LogisticRegression(): 
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
        