# import numpy as np

# class LogisticRegressionScratch:
#     """Logistic Regression classifier implemented from scratch."""
#     def __init__(self, lr=0.01, n_iter=1000, fit_intercept=True, verbose=False, l2_penalty=0.0, solver='newton', class_weight='balanced', tol=1e-6):
#         self.lr = lr
#         self.n_iter = n_iter
#         self.fit_intercept = fit_intercept
#         self.verbose = verbose
#         self.l2_penalty = l2_penalty
#         self.solver = solver
#         self.class_weight = class_weight
#         self.tol = tol

#     def __add_intercept(self, X):
#         intercept = np.ones((X.shape[0], 1))
#         return np.concatenate((intercept, X), axis=1)

#     def __sigmoid(self, z):
#         return 1 / (1 + np.exp(-z))

#     def fit(self, X, y):
#         if self.fit_intercept:
#             X = self.__add_intercept(X)
#         # compute sample weights for imbalance
#         if self.class_weight == 'balanced':
#             n0 = np.sum(y == 0)
#             n1 = np.sum(y == 1)
#             w0 = len(y) / (2 * n0)
#             w1 = len(y) / (2 * n1)
#             sample_weights = np.where(y == 1, w1, w0)
#         elif isinstance(self.class_weight, dict):
#             sample_weights = np.array([self.class_weight[label] for label in y])
#         else:
#             sample_weights = np.ones_like(y)
#         den = sample_weights.sum()
#         # early stopping tracker
#         prev_loss = None
#         # initialize parameters, seed intercept to log-odds of positive rate
#         self.theta = np.zeros(X.shape[1])
#         if self.fit_intercept:
#             p = np.clip(np.mean(y), 1e-5, 1 - 1e-5)
#             self.theta[0] = np.log(p / (1 - p))
#         for i in range(self.n_iter):
#             z = np.dot(X, self.theta)
#             h = self.__sigmoid(z)
#             gradient = X.T.dot((h - y) * sample_weights) / den
#             if self.l2_penalty:
#                 reg = (self.l2_penalty / den) * self.theta
#                 reg[0] = 0
#                 gradient += reg
#             if self.solver == 'newton':
#                 # compute Hessian
#                 S = h * (1 - h) * sample_weights / den
#                 X_weighted = X * S[:, np.newaxis]
#                 H = X.T.dot(X_weighted)
#                 # add L2 Hessian
#                 H += (self.l2_penalty / den) * np.eye(X.shape[1])
#                 H[0, 0] -= (self.l2_penalty / den)
#                 delta = np.linalg.solve(H, gradient)
#                 self.theta -= delta
#             else:
#                 self.theta -= self.lr * gradient
#             # compute loss for early stopping
#             loss = -((sample_weights * (y * np.log(h + 1e-15) + (1 - y) * np.log(1 - h + 1e-15))).sum() / den)
#             loss += (self.l2_penalty / (2 * den)) * np.sum(self.theta[1:]**2)
#             if self.verbose and i % max(1, self.n_iter // 10) == 0:
#                 print(f"Iteration {i}/{self.n_iter}: loss {loss:.4f}")
#             if prev_loss is not None and abs(prev_loss - loss) < self.tol:
#                 if self.verbose:
#                     print(f"Early stopping at iter {i}, Δloss={abs(prev_loss - loss):.2e}")
#                 break
#             prev_loss = loss
#         return self

#     def predict_proba(self, X):
#         if self.fit_intercept:
#             X = self.__add_intercept(X)
#         return self.__sigmoid(np.dot(X, self.theta))

#     def predict(self, X, threshold=0.5):
#         return (self.predict_proba(X) >= threshold).astype(int)
import numpy as np
from scipy import linalg

class LogisticRegressionScratch:
    """Enhanced Logistic Regression classifier for imbalanced data."""
    def __init__(self, lr=0.01, n_iter=1000, fit_intercept=True, verbose=False, 
                 l2_penalty=0.0, solver='newton', class_weight='balanced', 
                 tol=1e-8, min_iter=50, early_stopping=True):
        self.lr = lr
        self.n_iter = n_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.l2_penalty = l2_penalty
        self.solver = solver
        self.class_weight = class_weight
        self.tol = tol  # Much stricter tolerance
        self.min_iter = min_iter  # Minimum iterations before early stopping
        self.early_stopping = early_stopping
        self.theta = None
        self.history = {"loss": [], "delta_loss": []}
        
    def __add_intercept(self, X):
        """Add intercept term to feature matrix."""
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def __sigmoid(self, z):
        """Numerically stable sigmoid function."""
        # Clip values to avoid overflow/underflow
        z = np.clip(z, -250, 250)
        
        # More stable implementation for large negative values
        pos_mask = z >= 0
        neg_mask = ~pos_mask
        
        result = np.zeros_like(z, dtype=float)
        result[pos_mask] = 1.0 / (1.0 + np.exp(-z[pos_mask]))
        
        # For negative values, use exp(x)/(1+exp(x)) to avoid underflow
        exp_val = np.exp(z[neg_mask])
        result[neg_mask] = exp_val / (1.0 + exp_val)
        
        return result
        
    def fit(self, X, y):
        """Fit logistic regression with improved optimization."""
        # Add intercept if needed
        if self.fit_intercept:
            X = self.__add_intercept(X)
            
        # Compute sample weights for class imbalance
        if self.class_weight == 'balanced':
            n0 = np.sum(y == 0)
            n1 = np.sum(y == 1)
            w0 = len(y) / (2 * n0)
            w1 = len(y) / (2 * n1)
            sample_weights = np.where(y == 1, w1, w0)
        elif isinstance(self.class_weight, dict):
            sample_weights = np.array([self.class_weight.get(label, 1.0) for label in y])
        else:
            sample_weights = np.ones_like(y)
            
        den = sample_weights.sum()
        
        # Initialize parameters with better strategy
        self.theta = np.zeros(X.shape[1])
        if self.fit_intercept:
            # Initialize intercept using weighted log-odds
            p = np.clip(np.sum(y * sample_weights) / den, 1e-5, 1 - 1e-5)
            self.theta[0] = np.log(p / (1 - p))
            
        # Early stopping trackers
        prev_loss = None
        best_loss = float('inf')
        best_theta = None
        no_improvement_count = 0
        
        # Training loop
        for i in range(self.n_iter):
            # Compute predictions
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            
            # Compute weighted gradient
            gradient = X.T.dot((h - y) * sample_weights) / den
            
            # Add L2 regularization gradient
            if self.l2_penalty > 0:
                reg = (self.l2_penalty / den) * self.theta
                reg[0] = 0  # Don't regularize intercept
                gradient += reg
                
            if self.solver == 'newton':
                try:
                    # Compute Hessian with better numerical stability
                    S = h * (1 - h) * sample_weights / den
                    X_weighted = X * np.sqrt(S)[:, np.newaxis]
                    H = X_weighted.T.dot(X_weighted)
                    
                    # Add L2 regularization to Hessian
                    if self.l2_penalty > 0:
                        H += (self.l2_penalty / den) * np.eye(X.shape[1])
                        H[0, 0] -= (self.l2_penalty / den)  # Don't regularize intercept
                    
                    # Add small value to diagonal for numerical stability
                    H += 1e-8 * np.eye(X.shape[1])
                    
                    # Solve for update using more stable method
                    try:
                        delta = linalg.cho_solve(linalg.cho_factor(H), gradient)
                    except:
                        # Fallback to standard solve if Cholesky fails
                        delta = np.linalg.solve(H, gradient)
                    
                    self.theta -= delta
                    
                except np.linalg.LinAlgError:
                    # Fall back to gradient descent if Newton fails
                    if self.verbose:
                        print(f"Warning: Newton method failed at iter {i}, using gradient descent")
                    self.theta -= self.lr * gradient
            else:
                # Gradient descent with adaptive learning rate
                current_lr = self.lr / (1 + 0.01 * i)
                self.theta -= current_lr * gradient
                
            # Compute loss for tracking
            eps = 1e-15  # Small value for log stability
            loss = -((sample_weights * (y * np.log(h + eps) + 
                                      (1 - y) * np.log(1 - h + eps))).sum() / den)
            
            if self.l2_penalty > 0:
                loss += (self.l2_penalty / (2 * den)) * np.sum(self.theta[1:]**2)
                
            self.history["loss"].append(loss)
            
            # Track best model
            if loss < best_loss:
                best_loss = loss
                best_theta = self.theta.copy()
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                
            # Print progress if verbose
            if self.verbose and i % max(1, self.n_iter // 10) == 0:
                print(f"Iteration {i}/{self.n_iter}: loss {loss:.6f}")
                
            # Early stopping check (much improved)
            delta_loss = 0 if prev_loss is None else abs(prev_loss - loss)
            self.history["delta_loss"].append(delta_loss)
            
            if self.early_stopping and prev_loss is not None:
                if i >= self.min_iter and (delta_loss < self.tol or no_improvement_count >= 5):
                    if self.verbose:
                        print(f"Early stopping at iter {i}, Δloss={delta_loss:.2e}")
                    break
                    
            prev_loss = loss
            
        # Use best model parameters
        if best_theta is not None:
            self.theta = best_theta
            
        return self
        
    def predict_proba(self, X):
        """Predict class probabilities."""
        if self.fit_intercept:
            X = self.__add_intercept(X)
        return self.__sigmoid(np.dot(X, self.theta))
        
    def predict(self, X, threshold=0.5):
        """Predict class labels with threshold."""
        return (self.predict_proba(X) >= threshold).astype(int)