import numpy as np
from collections import Counter

class DecisionTreeScratch:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, 
                 max_features='sqrt', class_weight=None, criterion='gini'):
        self.max_depth = max_depth if max_depth is not None else float('inf')
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.class_weight = class_weight
        self.criterion = criterion  # Added support for multiple criteria
        self.tree = None
        self.feature_importances_ = None  # Added feature importance tracking

    def fit(self, X, y):
        # Calculate class weights if needed
        self.class_weights_ = self._compute_class_weights(y)
        
        # Compute sample weights based on class weights
        sample_weight = np.ones_like(y, dtype=float)
        for class_val, weight in self.class_weights_.items():
            sample_weight[y == class_val] = weight
            
        self.n_classes_ = len(np.unique(y))
        self.n_features_ = X.shape[1]
        self.feature_importances_ = np.zeros(self.n_features_)
        self.tree = self._grow_tree(X, y, sample_weight)
        
        # Normalize feature importances
        if np.sum(self.feature_importances_) > 0:
            self.feature_importances_ /= np.sum(self.feature_importances_)
            
        return self

    def _compute_class_weights(self, y):
        """Compute class weights based on class frequencies."""
        if self.class_weight == 'balanced':
            # Balanced weighting: inverse of frequency
            class_counts = Counter(y)
            total_samples = len(y)
            weights = {}
            for class_val, count in class_counts.items():
                weights[class_val] = total_samples / (len(class_counts) * count)
            return weights
        elif isinstance(self.class_weight, dict):
            return self.class_weight
        else:
            # Default: equal weights
            return {class_val: 1.0 for class_val in np.unique(y)}

    def _gini(self, y, sample_weight):
        """Weighted Gini impurity calculation."""
        if len(y) == 0:
            return 0
        
        # Get weighted proportions
        total_weight = np.sum(sample_weight)
        if total_weight == 0:
            return 0
            
        weighted_proportions = []
        for c in np.unique(y):
            mask = (y == c)
            class_weight = np.sum(sample_weight[mask])
            weighted_proportions.append((class_weight / total_weight) ** 2)
            
        return 1.0 - sum(weighted_proportions)

    def _entropy(self, y, sample_weight):
        """Weighted entropy calculation."""
        if len(y) == 0:
            return 0
            
        total_weight = np.sum(sample_weight)
        if total_weight == 0:
            return 0
            
        entropy = 0
        for c in np.unique(y):
            mask = (y == c)
            class_weight = np.sum(sample_weight[mask])
            p = class_weight / total_weight
            if p > 0:
                entropy -= p * np.log2(p)
                
        return entropy

    def _criterion_func(self, y, sample_weight):
        """Apply the selected split criterion."""
        if self.criterion == 'gini':
            return self._gini(y, sample_weight)
        elif self.criterion == 'entropy':
            return self._entropy(y, sample_weight)
        else:
            return self._gini(y, sample_weight)  # Default to Gini

    def _grow_tree(self, X, y, sample_weight, depth=0):
        """Build the decision tree recursively with weighted samples."""
        num_samples, num_features = X.shape
        
        # Calculate weighted class distribution
        classes, class_weights = np.unique(y, return_counts=False), []
        for c in classes:
            class_weights.append(np.sum(sample_weight[y == c]))
            
        # Determine predicted class (weighted majority)
        predicted_class = classes[np.argmax(class_weights)]
        
        # Create a leaf node by default
        node = {'type': 'leaf', 'class': predicted_class, 'samples': num_samples, 
                'weighted_samples': np.sum(sample_weight)}
        
        # Check stopping conditions
        if (depth >= self.max_depth or 
            num_samples < self.min_samples_split or 
            len(np.unique(y)) == 1):
            return node
            
        # Feature subsampling
        if self.max_features == 'sqrt':
            m = max(1, int(np.sqrt(num_features)))
        elif self.max_features == 'log2':
            m = max(1, int(np.log2(num_features)))
        elif isinstance(self.max_features, int):
            m = min(self.max_features, num_features)
        elif isinstance(self.max_features, float):
            m = max(1, int(self.max_features * num_features))
        else:
            m = num_features
            
        feat_idxs = np.random.choice(num_features, m, replace=False)
        
        # Find best split
        best_criterion = float('inf')
        best_feature, best_thresh = None, None
        
        # Track node purity for feature importance
        node_impurity = self._criterion_func(y, sample_weight)
        
        for feature_idx in feat_idxs:
            # Use percentiles for better threshold selection
            percentiles = np.linspace(5, 95, 10)
            thresholds = np.percentile(X[:, feature_idx], percentiles)
            
            for thr in thresholds:
                left_idx = X[:, feature_idx] < thr
                right_idx = ~left_idx
                
                # Ensure minimum samples in each leaf
                if (np.sum(left_idx) < self.min_samples_leaf or 
                    np.sum(right_idx) < self.min_samples_leaf):
                    continue
                    
                # Calculate weighted criterion
                left_weight = np.sum(sample_weight[left_idx])
                right_weight = np.sum(sample_weight[right_idx])
                total_weight = left_weight + right_weight
                
                if total_weight == 0:
                    continue
                    
                criterion_left = self._criterion_func(y[left_idx], sample_weight[left_idx])
                criterion_right = self._criterion_func(y[right_idx], sample_weight[right_idx])
                
                weighted_criterion = (left_weight * criterion_left + 
                                     right_weight * criterion_right) / total_weight
                
                if weighted_criterion < best_criterion:
                    best_criterion = weighted_criterion
                    best_feature = feature_idx
                    best_thresh = thr
        
        # If no valid split found, return leaf node
        if best_feature is None:
            return node
            
        # Update feature importance
        # Decrease in impurity = feature importance
        impurity_decrease = node_impurity - best_criterion
        self.feature_importances_[best_feature] += impurity_decrease * np.sum(sample_weight)
        
        # Build subtrees
        left_idx = X[:, best_feature] < best_thresh
        right_idx = ~left_idx
        
        left = self._grow_tree(X[left_idx], y[left_idx], sample_weight[left_idx], depth + 1)
        right = self._grow_tree(X[right_idx], y[right_idx], sample_weight[right_idx], depth + 1)
        
        return {
            'type': 'node', 
            'feature': best_feature, 
            'threshold': best_thresh,
            'impurity_decrease': impurity_decrease,
            'samples': num_samples,
            'weighted_samples': np.sum(sample_weight),
            'left': left, 
            'right': right
        }

    def _predict_sample(self, x, node):
        """Predict single sample."""
        if node['type'] == 'leaf':
            return node['class']
            
        if x[node['feature']] < node['threshold']:
            return self._predict_sample(x, node['left'])
            
        return self._predict_sample(x, node['right'])

    def predict(self, X):
        """Predict class labels for samples in X."""
        return np.array([self._predict_sample(x, self.tree) for x in X])
        
    def predict_proba(self, X):
        """Predict class probabilities for binary classification."""
        # This implementation works for binary classification only
        predictions = self.predict(X)
        return predictions.astype(float)


class RandomForestScratch:
    """Enhanced Random Forest classifier with improved handling for imbalanced data."""
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, 
                 min_samples_leaf=1, max_features='sqrt', bootstrap=True,
                 class_weight='balanced', criterion='gini', random_state=None,
                 stratify_bootstrap=True):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf  # Added parameter
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.class_weight = class_weight
        self.criterion = criterion
        self.random_state = random_state
        self.stratify_bootstrap = stratify_bootstrap  # Added stratification option
        self.trees = []
        self.feature_importances_ = None

    def fit(self, X, y):
        """Fit random forest to training data."""
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        self.trees = []
        n_samples, n_features = X.shape
        self.feature_importances_ = np.zeros(n_features)
        
        # Store classes for stratification
        self.classes_ = np.unique(y)
        
        # Progress indicator for verbose output
        step_size = max(1, self.n_estimators // 10)
        
        for i in range(self.n_estimators):
            # Progress indicator
            if (i + 1) % step_size == 0 or i + 1 == self.n_estimators:
                print(f"Training tree {i+1}/{self.n_estimators}")
                
            if self.bootstrap:
                if self.stratify_bootstrap:
                    # Stratified bootstrap sampling
                    bootstrap_indices = self._stratified_bootstrap(y, n_samples)
                else:
                    # Regular bootstrap sampling
                    bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
                    
                X_sample = X[bootstrap_indices]
                y_sample = y[bootstrap_indices]
            else:
                X_sample = X
                y_sample = y
                
            # Train tree
            tree = DecisionTreeScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                class_weight=self.class_weight,
                criterion=self.criterion
            )
            
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
            
            # Accumulate feature importances
            if tree.feature_importances_ is not None:
                self.feature_importances_ += tree.feature_importances_
                
        # Normalize feature importances
        if np.sum(self.feature_importances_) > 0:
            self.feature_importances_ /= len(self.trees)
            
        return self
        
    def _stratified_bootstrap(self, y, n_samples):
        """Create a bootstrap sample that preserves class distribution."""
        bootstrap_indices = []
        
        # Sample from each class with same proportion as original
        for cls in self.classes_:
            cls_indices = np.where(y == cls)[0]
            n_cls_samples = len(cls_indices)
            
            # Calculate how many samples to take from this class
            cls_sample_size = int(n_samples * (n_cls_samples / len(y)))
            
            # Ensure at least one sample per class
            cls_sample_size = max(1, cls_sample_size)
            
            # Sample with replacement from this class
            cls_bootstrap = np.random.choice(cls_indices, cls_sample_size, replace=True)
            bootstrap_indices.extend(cls_bootstrap)
            
        # Shuffle the indices
        np.random.shuffle(bootstrap_indices)
        
        # Adjust to exactly n_samples
        if len(bootstrap_indices) > n_samples:
            bootstrap_indices = bootstrap_indices[:n_samples]
        elif len(bootstrap_indices) < n_samples:
            # Fill any remaining slots with random samples
            remaining = n_samples - len(bootstrap_indices)
            additional = np.random.choice(len(y), remaining, replace=True)
            bootstrap_indices.extend(additional)
            
        return np.array(bootstrap_indices)

    def predict(self, X):
        """Predict class labels using majority voting."""
        # Collect predictions from all trees
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        
        # Use weighted voting
        weighted_votes = np.zeros((X.shape[0], len(self.classes_)))
        
        for i, cls in enumerate(self.classes_):
            # Count votes for each class
            for tree_pred in tree_preds:
                weighted_votes[:, i] += (tree_pred == cls)
                
        # Return class with most votes
        return self.classes_[np.argmax(weighted_votes, axis=1)]

    def predict_proba(self, X):
        """Predict class probabilities as proportion of votes."""
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        
        # For binary classification, return probability of positive class
        if len(self.classes_) == 2:
            # Count trees predicting class 1 (typically the positive class)
            pos_class_idx = np.where(self.classes_ == 1)[0][0]
            proba = np.mean(tree_preds == self.classes_[pos_class_idx], axis=0)
            
            # Return calibrated probabilities
            return self._calibrate_probabilities(proba)
        else:
            # For multi-class, return probability for each class
            proba = np.zeros((X.shape[0], len(self.classes_)))
            
            for i, cls in enumerate(self.classes_):
                proba[:, i] = np.mean(tree_preds == cls, axis=0)
                
            return proba
            
    def _calibrate_probabilities(self, proba):
        """Apply a simple calibration to probabilities to avoid extreme values."""
        # Clip probabilities to avoid 0 and 1
        epsilon = 1e-15
        proba = np.clip(proba, epsilon, 1 - epsilon)
        
        # Apply a simple logistic calibration to spread out the probabilities
        # This helps with decision threshold optimization
        calibrated = 1 / (1 + np.exp(-3 * (proba - 0.5)))
        
        return calibrated