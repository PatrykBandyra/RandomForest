# Author: Patryk Bandyra

import random
from dataclasses import dataclass
from typing import Union, Optional, Tuple, List

import numpy as np

from src import AttributeType


@dataclass
class Node:
    """
    Decision node or leaf in a decision tree
    """
    feature_index: Optional[int] = None  # If decision node, split is made using feature with this index
    threshold: Optional[Union[float, int]] = None
    left: Optional['Node'] = None
    right: Optional['Node'] = None
    info_gain: Optional[float] = None
    value: Optional[int] = None  # If leaf node, it stores class


class DecisionTree:
    """
    Decision tree:
    - each node can only have 2 children
    - it supports both continuous and categorical attributes
    """
    INFO_GAIN: str = 'info_gain'
    DATASET_LEFT: str = 'dataset_left'
    DATASET_RIGHT: str = 'dataset_right'
    FEATURE_INDEX: str = 'feature_index'
    THRESHOLD: str = 'threshold'
    INF: str = 'inf'
    GINI: str = 'gini'

    TOP_SPLITS_PCT: float = 0.5  # Hiper parameter - how many pct of best splits to choose in modified algorithm
    TOP_N_SPLITS_TOURNAMENT: int = 2  # How many randomly selected top splits take part in a tournament

    def __init__(self, dataset_attributes_types: List[AttributeType], min_samples_split: int = 2, max_depth: int = 2,
                 use_modified_algo: bool = True):
        self.root: Optional[Node] = None

        self.dataset_attributes_types: List[AttributeType] = dataset_attributes_types
        self.min_samples_split: int = min_samples_split  # The minimal number of samples in dataset to make a split
        self.max_depth: int = max_depth  # Maximal depth of a tree
        self.use_modified_algo: bool = use_modified_algo  # Flag allowing to choose between classic and modified algo

    def build_tree(self, dataset: np.ndarray, curr_depth: int = 0) -> Node:
        features: np.ndarray = dataset[:, :-1]
        targets: np.ndarray = dataset[:, -1]
        num_samples, num_features = np.shape(features)

        # Splitting until stopping conditions are met
        if num_samples >= self.min_samples_split and curr_depth <= self.max_depth:
            best_split: dict = self.get_split(dataset, num_features) if self.use_modified_algo else self.get_best_split(
                dataset, num_features)
            if best_split[DecisionTree.INFO_GAIN] > 0:
                left_subtree: Node = self.build_tree(best_split[DecisionTree.DATASET_LEFT], curr_depth + 1)
                right_subtree: Node = self.build_tree(best_split[DecisionTree.DATASET_RIGHT], curr_depth + 1)
                return Node(best_split[DecisionTree.FEATURE_INDEX], best_split[DecisionTree.THRESHOLD], left_subtree,
                            right_subtree, best_split[DecisionTree.INFO_GAIN])

        # Leaf node
        leaf_value: int = self.calculate_leaf_value(targets)
        return Node(value=leaf_value)

    def get_split(self, dataset: np.ndarray, num_features: int) -> dict:
        """
        Modified algorithm - chooses split in a following way:
        1. Calculate information gain for all possible splits
        2. Select best X% splits (X - TOP_SPLITS_PCT)
        3. From selected top splits select randomly 2
        4. Return better split from those 2 selected
        """
        splits_list = []

        for feature_index in range(num_features):
            feature_values: np.ndarray = dataset[:, feature_index]
            possible_thresholds: np.array = np.unique(feature_values)

            max_info_gain_for_feature = -float(DecisionTree.INF)
            best_split_for_feature = {}

            for threshold in possible_thresholds:
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    target: np.ndarray = dataset[:, -1]
                    target_left: np.ndarray = dataset_left[:, -1]
                    target_right: np.ndarray = dataset_right[:, -1]
                    curr_info_gain: float = self.information_gain(target, target_left, target_right, DecisionTree.GINI)
                    if curr_info_gain > max_info_gain_for_feature:
                        best_split_for_feature[DecisionTree.FEATURE_INDEX] = feature_index
                        best_split_for_feature[DecisionTree.THRESHOLD] = threshold
                        best_split_for_feature[DecisionTree.DATASET_LEFT] = dataset_left
                        best_split_for_feature[DecisionTree.DATASET_RIGHT] = dataset_right
                        best_split_for_feature[DecisionTree.INFO_GAIN] = curr_info_gain
                        max_info_gain_for_feature = curr_info_gain
            if best_split_for_feature:  # Check if dict is empty
                splits_list.append(best_split_for_feature)

        # Select top X% from splits list
        splits_to_choose_num: int = round(len(splits_list) * DecisionTree.TOP_SPLITS_PCT)
        if splits_to_choose_num == 1:  # Not enough features left
            return splits_list[0]

        top_splits_list = []
        for i in range(splits_to_choose_num):
            max_info_gain_split = max(splits_list, key=lambda split: split[DecisionTree.INFO_GAIN])
            top_splits_list.append(max_info_gain_split)
            splits_list.remove(max_info_gain_split)

        # Select random 2 top splits
        splits_tournament_list = []
        for i in range(DecisionTree.TOP_N_SPLITS_TOURNAMENT):
            chosen_split = random.choice(top_splits_list)
            splits_tournament_list.append(chosen_split)
            top_splits_list.remove(chosen_split)

        # Tournament - select best split
        best_split = max(splits_tournament_list, key=lambda split: split[DecisionTree.INFO_GAIN])

        return best_split

    def get_best_split(self, dataset: np.ndarray, num_features: int) -> dict:
        """Classic algorithm - chooses split with the greatest information gain."""
        best_split = {}
        max_info_gain = -float(DecisionTree.INF)

        for feature_index in range(num_features):
            feature_values: np.ndarray = dataset[:, feature_index]
            possible_thresholds: np.array = np.unique(feature_values)

            for threshold in possible_thresholds:
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    target: np.ndarray = dataset[:, -1]
                    target_left: np.ndarray = dataset_left[:, -1]
                    target_right: np.ndarray = dataset_right[:, -1]
                    curr_info_gain: float = self.information_gain(target, target_left, target_right, DecisionTree.GINI)
                    if curr_info_gain > max_info_gain:
                        best_split[DecisionTree.FEATURE_INDEX] = feature_index
                        best_split[DecisionTree.THRESHOLD] = threshold
                        best_split[DecisionTree.DATASET_LEFT] = dataset_left
                        best_split[DecisionTree.DATASET_RIGHT] = dataset_right
                        best_split[DecisionTree.INFO_GAIN] = curr_info_gain
                        max_info_gain = curr_info_gain
        return best_split

    def split(self, dataset: np.ndarray, feature_index: int, threshold: float) -> Tuple[np.array, np.array]:
        """Splits dataset into left and right subsets. Split is made differently depending on attribute type."""
        if self.dataset_attributes_types[feature_index] == AttributeType.QUANTITATIVE:
            dataset_left = np.array([row for row in dataset if row[feature_index] <= threshold])
            dataset_right = np.array([row for row in dataset if row[feature_index] > threshold])
        else:
            dataset_left = np.array([row for row in dataset if row[feature_index] == threshold])
            dataset_right = np.array([row for row in dataset if row[feature_index] != threshold])
        return dataset_left, dataset_right

    def information_gain(self, parent: np.ndarray, child_l: np.ndarray, child_r: np.ndarray, mode='entropy') -> float:
        """Calculates weighted information gain using selected mode. Modes: entropy or gini (index)."""
        weight_l: float = len(child_l) / len(parent)
        weight_r: float = len(child_r) / len(parent)
        if mode == DecisionTree.GINI:
            gain = self.gini_index(parent) - (weight_l * self.gini_index(child_l) + weight_r * self.gini_index(child_r))
        else:
            gain = self.entropy(parent) - (weight_l * self.entropy(child_l) + weight_r * self.entropy(child_r))
        return gain

    @staticmethod
    def entropy(targets: np.ndarray) -> float:
        class_labels: np.ndarray = np.unique(targets)
        entropy: float = 0
        for class_label in class_labels:
            cls_probability: float = len(targets[targets == class_label]) / len(targets)
            entropy += -cls_probability * np.log2(cls_probability)
        return entropy

    @staticmethod
    def gini_index(targets: np.ndarray) -> float:
        class_labels: np.ndarray = np.unique(targets)
        gini: float = 0
        for class_label in class_labels:
            cls_probability: float = len(targets[targets == class_label]) / len(targets)
            gini += cls_probability ** 2
        return 1 - gini

    @staticmethod
    def calculate_leaf_value(targets: np.ndarray) -> int:
        """Calculates which label should be in a leaf."""
        targets: list = list(targets)
        return max(targets, key=targets.count)

    def print_tree(self, tree: Optional[Node] = None, indent: str = ' ') -> None:
        if not tree:
            tree: Node = self.root

        if tree.value is not None:
            print(tree.value)
        else:
            print(f'X_{tree.feature_index}<={tree.threshold}?{tree.info_gain}')
            print('%sleft:' % indent, end='')
            self.print_tree(tree.left, indent + indent)
            print('%sright:' % indent, end='')
            self.print_tree(tree.right, indent + indent)

    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
        dataset: np.ndarray = np.concatenate((features, targets), axis=1)
        self.root = self.build_tree(dataset)

    def fit_whole(self, dataset: np.ndarray) -> None:
        self.root = self.build_tree(dataset)

    def predict(self, features: np.ndarray) -> List[int]:
        predictions: List[int] = [self.make_prediction(x, self.root) for x in features]
        return predictions

    def make_prediction(self, x: np.ndarray, tree: Node) -> int:
        if tree.value is not None:  # Leaf node
            return tree.value
        feature_val: float = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)
