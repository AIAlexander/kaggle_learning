import pandas as pd


def read_wine_data(path="../../data/winemag-data-130k-v2.csv"):
    # show the five data
    pd.set_option('max_rows', 5)
    return pd.read_csv(path, index_col=0)