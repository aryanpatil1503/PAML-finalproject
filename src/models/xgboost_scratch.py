import numpy as np
from sklearn.tree import DecisionTreeRegressor


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class XGBoostScratch:
    def __init__(self,
                 n_estimators=100,
                 learning_rate=0.1,
                 max_depth=3,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 subsample=1.0,
                 random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.random_state = random_state
        self.trees = []
        self.init_pred = 0.0

    def fit(self, X, y):
        n_samples = X.shape[0]
        # initial raw score as log-odds
        p = np.clip(np.mean(y), 1e-5, 1 - 1e-5)
        self.init_pred = np.log(p / (1 - p))
        # running prediction
        Fm = np.full(n_samples, self.init_pred)
        rng = np.random.RandomState(self.random_state)

        for m in range(self.n_estimators):
            # probability and residual (gradient)
            Pm = _sigmoid(Fm)
            residual = y - Pm
            # subsample
            if self.subsample < 1.0:
                idx = rng.choice(n_samples,
                                  int(self.subsample * n_samples),
                                  replace=False)
                X_train, r_train = X[idx], residual[idx]
            else:
                X_train, r_train = X, residual
            # fit regression tree on residuals
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state
            )
            tree.fit(X_train, r_train)
            self.trees.append(tree)
            # update ensemble prediction
            update = tree.predict(X)
            Fm += self.learning_rate * update

    def predict_proba(self, X):
        # start from initial
        Fm = np.full(X.shape[0], self.init_pred)
        for tree in self.trees:
            Fm += self.learning_rate * tree.predict(X)
        proba = _sigmoid(Fm)
        # return two-column array [[1-p, p]]
        return np.vstack((1 - proba, proba)).T

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)
