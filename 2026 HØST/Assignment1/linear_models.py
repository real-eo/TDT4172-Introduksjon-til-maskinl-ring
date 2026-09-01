import numpy as np

class LinearRegression():
    def __init__(self, lr=0.001, n_iterations=1000):
        self.weights = None
        self.bias = None
        
        self.lr = lr
        self.n_iterations = n_iterations
        
        self.loss_history = []
        
    def fit(self, X, y):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats
        """
        # ====================================
        # YOUR CODE GOES HERE
        # ====================================
        raise NotImplementedError("LinearRegression.fit is not implemented yet.")
    
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
        # ====================================
        # YOUR CODE GOES HERE
        # ====================================
        raise NotImplementedError("LinearRegression.predict is not implemented yet.")
    
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