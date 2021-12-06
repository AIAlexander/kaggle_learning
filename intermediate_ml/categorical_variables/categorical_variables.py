"""

A categorical variable takes only a limited number of values
e.g. a survey that asks how often you eat breakfast and provides four options:
    "Never", "Rarely", "Most days", "Every day"
In this case the data is categorical, because responses fall into a fixed set of categories

Get an error if try to plug these variable into most machine learning models in Python without processing them first,
so there are three approach to prepare the categorical data.

"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OrdinalEncoder


def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)


def drop_categorical_variables():
    """
    Simply remove them from the dataset.
    This approach will only work well if the columns did not contain useful information
    :return:
    """

    # read the data
    data = pd.read_csv('../../data/melb_data.csv')

    # Separate target from predictors
    y = data.Price
    X = data.drop(['Price'], axis=1)

    # Divide data into training and validation subsets
    X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

    col_with_missing = [col for col in X_train_full if X_train_full[col].isnull().any()]
    X_train_full.drop(col_with_missing, axis=1, inplace=True)
    X_valid_full.drop(col_with_missing, axis=1, inplace=True)

    # select categorical columns with relatively low cardinally (convenient but arbitrary)
    low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and X_train_full[cname].dtype == "object"]

    # select the numerical colums
    numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

    my_cols = low_cardinality_cols + numerical_cols
    X_train = X_train_full[my_cols].copy()
    X_valid = X_valid_full[my_cols].copy()

    # get list of categorical variables
    s = (X_train.dtypes == 'object')
    object_cols = list(s[s].index)
    print("Categorical columns:")
    print(object_cols)

    drop_X_train = X_train.select_dtypes(exclude=['object'])
    drop_X_valid = X_valid.select_dtypes(exclude=['object'])

    print("MAE from Approach 1 (Drop categorical variables):")
    print(score_dataset(drop_X_train, drop_X_valid, y_train, y_valid))


def ordinal_encoding():
    """
    Ordinal encoding assigns each categorical variables to a different integer
    For tree-based model(like decision trees and random forests), ordinal encoding work well.
    :return:
    """

    # read the data
    data = pd.read_csv('../../data/melb_data.csv')

    # Separate target from predictors
    y = data.Price
    X = data.drop(['Price'], axis=1)

    # Divide data into training and validation subsets
    X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

    col_with_missing = [col for col in X_train_full if X_train_full[col].isnull().any()]
    X_train_full.drop(col_with_missing, axis=1, inplace=True)
    X_valid_full.drop(col_with_missing, axis=1, inplace=True)

    # select categorical columns with relatively low cardinally (convenient but arbitrary)
    low_cardinality_cols = [cname for cname in X_train_full.columns if
                            X_train_full[cname].nunique() < 10 and X_train_full[cname].dtype == "object"]

    # select the numerical colums
    numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

    my_cols = low_cardinality_cols + numerical_cols
    X_train = X_train_full[my_cols].copy()
    X_valid = X_valid_full[my_cols].copy()

    # get list of categorical variables
    s = (X_train.dtypes == 'object')
    object_cols = list(s[s].index)
    print("Categorical columns:")
    print(object_cols)

    # Make copy to avoid changing original data
    label_X_train = X_train.copy()
    label_X_valid = X_valid.copy()

    # Apply ordinal encoder to each column with categorical data
    ordinal_encoding = OrdinalEncoder()
    label_X_train[object_cols] = ordinal_encoding.fit_transform(X_train[object_cols])
    label_X_valid[object_cols] = ordinal_encoding.transform(X_valid[object_cols])

    print("MAE from Approach 2 (Ordinal Encoding):")
    print(score_dataset(label_X_train, label_X_valid, y_train, y_valid))


def one_hot_encoding():
    """
    One-hot encoding creates new columns indicating the presence of each possible value in the original data.
    e.g. Color is a categorical variable with three categories 'Red', 'Yellow' and 'Green'
    One-hot encoding
    :return:
    """


if __name__ == '__main__':
    # drop_categorical_variables()
    ordinal_encoding()