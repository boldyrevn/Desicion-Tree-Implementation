from DecisionTree import DecisionTree
import numpy as np
import pandas as pd


class RandomForest:
    def __init__(self, n_features=None, max_depth=100, min_samples_leaf=2,
                 min_impurity_decrease=0.0, n_trees=10, random_state=None):
        self.n_features = n_features
        self.max_depth = max_depth
        self.min_samples_leaf = 2
        self.min_impurity_decrease = 0.0
        self.n_trees = n_trees
        self.trees = []
        self.random_state = random_state

    @staticmethod
    def _most_common_label(y):
        labels, counts = np.unique(y, return_counts=True)
        idx = np.argmax(counts)
        return labels[idx]

    def _bootstrap(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.RandomState(self.random_state).choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def fit(self, X, y):
        if type(X) == pd.DataFrame:
            X = X.to_numpy()
            y = y.to_numpy()

        self.trees = []
        for _ in range(self.n_trees):
            new_tree = DecisionTree(min_samples_leaf=self.min_samples_leaf, max_depth=self.max_depth,
                                    n_features=self.n_features,min_impurity_decrease=self.min_impurity_decrease,
                                    random_state=self.random_state)

            X_samples, y_samples = self._bootstrap(X, y)
            new_tree.fit(X_samples, y_samples)
            self.trees.append(new_tree)

    def predict(self, X):
        preds = np.array([tree.predict(X) for tree in self.trees]).T
        voted_preds = np.array([RandomForest._most_common_label(y) for y in preds])
        return voted_preds

    def predict_proba(self, X):
        preds = np.array([tree.predict_proba(X) for tree in self.trees]).T
        mean_preds = np.array([np.mean(y) for y in preds])
        return mean_preds
