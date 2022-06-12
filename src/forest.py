# Author: Oskar Bartosz

import numpy as np
import pandas as pd


class RandomForest:
    def __init__(self, tree, tree_args: tuple, n_trees: int = 10) -> None:
        self.trees = []
        for tree_id in range(n_trees):
            self.trees.append(tree(*tree_args))

    def fit(self, data: pd.DataFrame, sample_size: int = None):
        if sample_size is None:
            sample_size = int(np.sqrt(data.shape[0]))
        for tree in self.trees:
            sample = data.sample(n=sample_size, random_state=1).to_numpy()
            tree.fit_whole(sample)

    def predict(self, features: np.ndarray):
        votes = np.ndarray((features.shape[0], 0))

        for tree in self.trees:
            votes = np.c_[votes, tree.predict(features)]

        results = np.array([])
        for row in votes:
            a, b = np.unique(row, return_counts=True)
            results = np.append(results, a[np.argmax(b)])

        return results
