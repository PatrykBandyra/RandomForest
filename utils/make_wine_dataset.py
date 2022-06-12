# Author: Patryk Bandyra

import pandas as pd
from sklearn.utils import shuffle


def main():
    red_df: pd.DataFrame = pd.read_csv('winequality-red.csv', delimiter=';')
    white_df: pd.DataFrame = pd.read_csv('winequality-white.csv', delimiter=';')

    red_df.insert(11, 'type', 'red')
    white_df.insert(11, 'type', 'white')

    all_df: pd.DataFrame = pd.concat([red_df, white_df])
    all_df = shuffle(all_df)

    all_df.to_csv('data/wine-quality.csv', index=False, sep=';')


def test_if_ok():
    pd.set_option('display.max_columns', None)
    df: pd.DataFrame = pd.read_csv('wine.csv', delimiter=';')
    print(df.head(5))


if __name__ == '__main__':
    main()
    test_if_ok()
