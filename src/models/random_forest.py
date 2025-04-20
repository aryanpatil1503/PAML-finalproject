import numpy as np

class DecisionTreeScratch:
    """A simple decision tree classifier using Gini impurity."""
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth if max_depth is not None else float('inf')
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.n_classes_ = len(np.unique(y))
        self.n_features_ = X.shape[1]
        self.tree = self._grow_tree(X, y)
        return self

    def _gini(self, y):
        m = len(y)
        if m == 0:
            return 0
        return 1.0 - sum((np.sum(y == c) / m) ** 2 for c in np.unique(y))

    def _grow_tree(self, X, y, depth=0):
        # create a leaf node
        num_samples, num_features = X.shape
        num_labels = [np.sum(y == c) for c in np.unique(y)]
        predicted_class = np.unique(y)[np.argmax(num_labels)]
        node = {'type': 'leaf', 'class': predicted_class}
        # stopping conditions
        if depth >= self.max_depth or num_samples < self.min_samples_split or len(np.unique(y)) == 1:
            return node
        # find best split
        best_gini = 1.0
        best_feature, best_thresh = None, None
        for feature_idx in range(num_features):
            thresholds = np.unique(X[:, feature_idx])
            for thr in thresholds:
                left_idx = X[:, feature_idx] < thr
                right_idx = ~left_idx
                if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
                    continue
                gini_left = self._gini(y[left_idx])
                gini_right = self._gini(y[right_idx])
                gini = (np.sum(left_idx) * gini_left + np.sum(right_idx) * gini_right) / num_samples
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature_idx
                    best_thresh = thr
        if best_feature is None:
            return node
        # build subtrees
        left_idx = X[:, best_feature] < best_thresh
        right_idx = ~left_idx
        left = self._grow_tree(X[left_idx], y[left_idx], depth + 1)
        right = self._grow_tree(X[right_idx], y[right_idx], depth + 1)
        return {'type': 'node', 'feature': best_feature, 'threshold': best_thresh, 'left': left, 'right': right}

    def _predict_sample(self, x, node):
        if node['type'] == 'leaf':
            return node['class']
        if x[node['feature']] < node['threshold']:
            return self._predict_sample(x, node['left'])
        return self._predict_sample(x, node['right'])

    def predict(self, X):
        return np.array([self._predict_sample(x, self.tree) for x in X])

class RandomForestScratch:
    """Random Forest classifier using bootstrap aggregation of decision trees."""
    def __init__(self, n_estimators=10, max_depth=None, min_samples_split=2, max_features='sqrt'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        n_samples, n_features = X.shape
        for _ in range(self.n_estimators):
            # bootstrap sample
            idxs = np.random.choice(n_samples, n_samples, replace=True)
            X_sample = X[idxs]
            y_sample = y[idxs]
            # select subset of features
            if self.max_features == 'sqrt':
                m = max(1, int(np.sqrt(n_features)))
            else:
                m = n_features
            feat_idxs = np.random.choice(n_features, m, replace=False)
            # train tree
            tree = DecisionTreeScratch(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.feature_indices = feat_idxs
            tree.fit(X_sample[:, feat_idxs], y_sample)
            self.trees.append(tree)
        return self

    def predict(self, X):
        # collect predictions
        tree_preds = np.array([tree.predict(X[:, tree.feature_indices]) for tree in self.trees])
        # majority vote
        preds = np.apply_along_axis(lambda x: np.bincount(x, minlength=2).argmax(), axis=0, arr=tree_preds)
        return preds

    def predict_proba(self, X):
        tree_preds = np.array([tree.predict(X[:, tree.feature_indices]) for tree in self.trees])
        # probability of class 1
        proba = np.mean(tree_preds, axis=0)
        return proba
