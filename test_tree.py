# Author: Patryk Bandyra

import random

from sklearn.metrics import accuracy_score

from src import DataLoader, DecisionTree


def test_on_only_numeric_dataset() -> None:
    dl = DataLoader.load_iris()
    data = dl.get_data()
    features_train, features_test, targets_train, targets_test = DataLoader.dataset_split(data)
    decision_tree = DecisionTree(dataset_attributes_types=dl.get_col_att_types(), min_samples_split=3, max_depth=3)
    decision_tree.fit(features_train, targets_train)
    decision_tree.print_tree()
    predictions = decision_tree.predict(features_test)
    print(f'Iris Accuracy: {accuracy_score(targets_test, predictions)}')


def test_on_mixed_dataset() -> None:
    dl = DataLoader.load_wine()
    data = dl.get_data()
    features_train, features_test, targets_train, targets_test = DataLoader.dataset_split(data)
    decision_tree = DecisionTree(dataset_attributes_types=dl.get_col_att_types(), min_samples_split=3, max_depth=3)
    decision_tree.fit(features_train, targets_train)
    decision_tree.print_tree()
    predictions = decision_tree.predict(features_test)
    print(f'Wine Accuracy: {accuracy_score(targets_test, predictions)}')


def main() -> None:
    random.seed(10)
    test_on_only_numeric_dataset()
    test_on_mixed_dataset()


if __name__ == "__main__":
    main()
