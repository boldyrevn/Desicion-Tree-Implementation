import numpy as np
import numpy.random
import pandas as pd


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None, proba=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.proba = proba

    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, min_samples_leaf=2, max_depth=100, n_features=None,
                 min_impurity_decrease=0.0, random_state=None):
        self._min_samples_leaf = min_samples_leaf
        self._max_depth = max_depth
        self._n_features = n_features
        self._root = None
        self._min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state

    @staticmethod
    def _most_common_label(y):
        labels, counts = np.unique(y, return_counts=True)
        idx = np.argmax(counts)
        return labels[idx], counts[idx]

    @staticmethod
    def _split(X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).ravel()
        right_idxs = np.argwhere(X_column > split_thresh).ravel()
        return left_idxs, right_idxs

    @staticmethod
    def _entropy(y):
        p = np.bincount(y) / len(y)
        p = p[p > 0]
        return np.sum(-p * np.log2(p))

    @staticmethod
    def _information_gain(X_column, threshold, y):
        # parent_entropy
        parent_entropy = DecisionTree._entropy(y)

        # create children
        left_idx, right_idx = DecisionTree._split(X_column, threshold)
        if len(right_idx) == 0 or len(left_idx) == 0:
            return 0

        # calculate weighted average children entropy
        y_left, y_right = y[left_idx], y[right_idx]
        n = len(y)
        n_l, n_r = len(y_left), len(y_right)
        e_l, e_r = DecisionTree._entropy(y_left), DecisionTree._entropy(y_right)
        children_entropy = (n_l / n) * e_l + (n_r / n) * e_r
        return parent_entropy - children_entropy

    @staticmethod
    def _best_split(X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None

        # find the best feature and threshold for splitting
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for thresh in thresholds:
                gain = DecisionTree._information_gain(X_column, thresh, y)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = thresh

        return split_idx, split_thresh, best_gain

    def _grow_tree(self, X, y, depth):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # check the stopping criteria
        if n_labels == 1 or depth >= self._max_depth or n_samples < self._min_samples_leaf:
            leaf_value, value_count = DecisionTree._most_common_label(y)
            leaf_proba = value_count / y.shape if leaf_value == 1 else 1 - value_count / y.shape
            return Node(value=leaf_value, proba=leaf_proba)

        # choose the pseudo random features
        feat_idxs = np.random.RandomState(self.random_state).choice(n_feats, self._n_features, replace=False)

        # find the best split
        best_feature, best_thresh, best_gain = DecisionTree._best_split(X, y, feat_idxs)
        if best_gain < self._min_impurity_decrease:
            leaf_value, value_count = DecisionTree._most_common_label(y)
            leaf_proba = value_count / y.shape if leaf_value == 1 else 1 - value_count / y.shape
            return Node(value=leaf_value, proba=leaf_proba)

        # create child nodes
        left_idx, right_idx = DecisionTree._split(X[:, best_feature], best_thresh)
        left_node = self._grow_tree(X[left_idx, :], y[left_idx], depth + 1)
        right_node = self._grow_tree(X[right_idx, :], y[right_idx], depth + 1)
        return Node(best_feature, best_thresh, left_node, right_node)

    def fit(self, X, y):
        if type(X) == pd.DataFrame:
            X = X.to_numpy()
            y = y.to_numpy()

        self._n_features = X.shape[1] if self._n_features is None else min(X.shape[1], self._n_features)
        self._root = self._grow_tree(X, y, 0)

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def _traverse_tree_proba(self, x, node):
        if node.is_leaf_node():
            return node.proba

        if x[node.feature] <= node.threshold:
            return self._traverse_tree_proba(x, node.left)
        return self._traverse_tree_proba(x, node.right)

    def predict(self, X):
        if type(X) == pd.DataFrame:
            X = X.to_numpy()
        return np.array([self._traverse_tree(x, self._root) for x in X]).ravel()

    def predict_proba(self, X):
        if type(X) == pd.DataFrame:
            X = X.to_numpy()
        return np.array([self._traverse_tree_proba(x, self._root) for x in X]).ravel()

