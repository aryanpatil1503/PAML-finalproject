import numpy as np

class LogisticRegressionScratch:
    """Logistic Regression classifier implemented from scratch."""
    def __init__(self, lr=0.01, n_iter=1000, fit_intercept=True, verbose=False, l2_penalty=0.0):
        self.lr = lr
        self.n_iter = n_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.l2_penalty = l2_penalty

    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        self.theta = np.zeros(X.shape[1])
        for i in range(self.n_iter):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            if self.l2_penalty:
                reg = (self.l2_penalty / y.size) * self.theta
                reg[0] = 0
                gradient += reg
            self.theta -= self.lr * gradient
            if self.verbose and i % max(1, self.n_iter // 10) == 0:
                loss = - (y * np.log(h + 1e-15) + (1 - y) * np.log(1 - h + 1e-15)).mean()
                loss += (self.l2_penalty / (2 * y.size)) * np.sum(self.theta[1:]**2)
                print(f"Iteration {i}/{self.n_iter}: loss {loss:.4f}")
        return self

    def predict_proba(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        return self.__sigmoid(np.dot(X, self.theta))

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
