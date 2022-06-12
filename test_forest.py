# Author: Oskar Bartosz

import random

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src import RandomForest, DataLoader, DecisionTree, make_raport


def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def test_uma_random_forest(dl: DataLoader, raport_title="UMA Random Forest", hiperparameters=(3, 3, 20, 50)):
    """
    Hiperparameters passed as tuple length 4:
    id 0 - max tree depth
    id 1 - max split
    id 2 - number of trees in forest
    id 3 - sample size for building single tree
    """
    fr: RandomForest
    tree_args = (dl.get_col_att_types(), hiperparameters[0], hiperparameters[1], True)
    fr = RandomForest(DecisionTree, tree_args=tree_args, n_trees=hiperparameters[2])

    train_data, test_data = DataLoader.split(dl.get_data(), 0.7)
    test_features, test_targets = DataLoader.separate_targets(test_data)
    test_targets = np.ravel(test_targets)

    fr.fit(train_data, hiperparameters[3])
    predictions = fr.predict(test_features)

    raport = make_raport(test_targets, predictions, raport_title)
    return raport


def test_scilearn_forest(dl: DataLoader, raport_title="SciLearn Random Forest"):
    train_data, test_data = DataLoader.split(dl.get_data(), 0.7)
    train_features, train_targets = DataLoader.separate_targets(train_data)
    train_targets = np.ravel(train_targets)
    test_features, test_targets = DataLoader.separate_targets(test_data)
    test_targets = np.ravel(test_targets)

    rfc = RandomForestClassifier(max_depth=3, random_state=0)
    rfc.fit(train_features, train_targets)
    predictions = np.ravel(rfc.predict(test_features))

    raport = make_raport(test_targets, predictions, raport_title)
    return raport


def cross_test(dl, measure='acc'):
    eval_map = {'acc': 0, 'prec': 1, 'rec': 2}
    tree_depth = np.array([4, 3])
    min_samples_splt = np.array([4, 3])
    num_trees = np.array([50, 30, 10])
    sample_size = np.array([40, 60, 80])  # iris
    # sample_size = np.array([1000, 500, 300]) #wine
    hiperparameters = cartesian_product(tree_depth, min_samples_splt, num_trees, sample_size)
    results = np.ndarray((0, 3))
    for row in hiperparameters:
        print(f"Test: {row}")
        res = np.expand_dims(np.array(test_uma_random_forest(dl, "cvw/" + str(row), row)[:-1]), axis=0)
        results = np.r_[results, res]
    eval_column = results[:, eval_map[measure]]
    print(f"Eval\n{eval_column}")
    return hiperparameters[np.argmax(eval_column)]


def main() -> None:
    random.seed(10)
    dl = DataLoader.load_iris()
    # dl = DataLoader.load_wine()
    test_scilearn_forest(dl, "SciLearn Random Forest")
    best_params = cross_test(dl, measure='prec')
    test_uma_random_forest(dl, f" UMA best: {best_params[0]} {best_params[1]} {best_params[2]} {best_params[3]}",
                           best_params)


if __name__ == "__main__":
    main()
