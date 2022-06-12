# Author: Oskar Bartosz

from enum import Enum
from typing import Union, Tuple, List

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class AttributeType(Enum):
    """
    Qualitative: nominal, ordinal, binary; can be represented as strings or numbers
    Quantitative: numeric, discrete, continuous; must be numbers
    Author: Patryk Bandyra
    """
    QUALITATIVE: str = 'qualitative'
    QUANTITATIVE: str = 'quantitative'

    @classmethod
    def values(cls):
        return list(map(lambda c: c.value, cls))


class DataLoader:
    def __init__(self, data: pd.DataFrame, att_types=None, label_names=None) -> None:
        self.data = data
        self.label_id = len(self.data.columns) - 1
        self.column_atribute_types = att_types
        self.label_names = label_names

    @classmethod
    def from_file(cls, path: str, with_header: bool = None):
        data = pd.read_csv(path, header=with_header)
        return cls(data)

    @classmethod
    def load_iris(cls):
        col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'type']
        data: pd.DataFrame = pd.read_csv('data/iris.csv', skiprows=1, header=None, names=col_names)
        label_names = data.iloc[:, -1].unique()
        data = cls.encode_nonnumeric(data)
        col_types: List[Union[AttributeType.values()]] = [AttributeType.QUANTITATIVE for _ in range(len(data.columns))]
        return cls(data, col_types, label_names)

    @classmethod
    def load_wine(cls):
        data: pd.DataFrame = pd.read_csv('data/wine.csv', delimiter=';')
        label_names = data.iloc[:, -1].unique()
        data = cls.encode_nonnumeric(data)
        col_types: List[Union[AttributeType.values()]] = [AttributeType.QUANTITATIVE for _ in
                                                          range(len(data.columns))] + [AttributeType.QUALITATIVE,
                                                                                       AttributeType.QUANTITATIVE]
        return cls(data, col_types, label_names)

    @classmethod
    def dataset_split(cls, data: pd.DataFrame, test_size=0.2, random_state=41) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        features: pd.DataFrame = data.iloc[:, :-1].values
        targets: pd.DataFrame = data.iloc[:, -1].values.reshape(-1, 1)
        return train_test_split(features, targets, test_size=test_size, random_state=random_state)

    def get_data(self):
        return self.data

    def get_labels(self):
        return self.label_names

    def get_col_att_types(self):
        return self.column_atribute_types

    def set_label_id(self, id: int) -> None:
        self.label_id = id
        return self

    @classmethod
    def split(cls, data, fraction):
        data = data.sample(frac=1, random_state=1).reset_index(drop=True)
        assert (fraction > 0 and fraction < 1)
        cut_id = int(data.shape[0] * fraction)
        return (data.iloc[:cut_id, :], data.iloc[cut_id:, :])

    @classmethod
    def separate_targets(cls, data):
        features: pd.DataFrame = data.iloc[:, :-1].values
        targets: pd.DataFrame = data.iloc[:, -1].values.reshape(-1, 1)
        return (features, targets)

    def change(self, data: pd.DataFrame) -> None:
        self.data = data

    def shuffle(self) -> pd.DataFrame:
        return self.data.sample(frac=1, random_state=1).reset_index(drop=True)

    def split_groups(self, kgroups: int) -> list:
        return np.array_split(self.shuffle(), kgroups)

    def group_by_label(self) -> dict:
        return {label: self.data[self.data.iloc[:, self.label_id] == label] for label in self.labels()}

    def labels(self, index: int = None):
        if index is None:
            return self.data.iloc[:, self.label_id].unique()
        else:
            return self.data.iloc[:, self.label_id].unique()[index]

    def unique(self, index: int):
        return self.data.iloc[:, index].unique()

    @classmethod
    def encode_nonnumeric(cls, data, onehot=False):
        le = LabelEncoder()
        for column in data.columns:
            if not is_numeric_dtype(data[column]):
                data[column] = le.fit_transform(data[column])
        return data

    def columns(self, index: int = None, avoid=[]) -> list:
        if index is None:
            return [item for item in self.data.columns if item not in avoid]
        else:
            return self.data.columns[index]

    def exploratory_analysis(self, col_id: int = None, search_for: str = None, with_label: bool = False):
        results = {}
        avoid_ = [self.label_id if not with_label else []]
        for col in self.columns(avoid=avoid_):
            results[col] = {}
            results[col]["mean"] = self.data.iloc[:, col].mean()
            results[col]["std"] = self.data.iloc[:, col].std()
            results[col]["min"] = self.data.iloc[:, col].min()
            results[col]["max"] = self.data.iloc[:, col].max()
        if col_id is None:
            return results
        else:
            if search_for is None:
                return results[col_id]
            else:
                return results[col_id][search_for]


if __name__ == "__main__":
    dl = DataLoader.load_iris()
    print(dl.data)
