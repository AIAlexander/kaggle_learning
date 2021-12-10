import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


def cross_validation():
    # Read the data
    train_data = pd.read_csv('../../data/house_price/train.csv', index_col='Id')
    test_data = pd.read_csv('../../data/house_price/test.csv', index_col='Id')

    # Remove rows with missing target, separate target from predictors
    train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
    y = train_data.SalePrice
    train_data.drop(['SalePrice'], axis=1, inplace=True)

    # Select numerical col
    num_cols = [cname for cname in train_data.columns
                if train_data[cname].dtype in ['int64', 'float64']]
    X = train_data[num_cols].copy()
    X_test = test_data[num_cols].copy()

    my_pipeline = Pipeline(steps=[
        ('preprocessor', SimpleImputer()),
        ('model', RandomForestRegressor(n_estimators=50, random_state=0))
    ])

    scores = -1 * cross_val_score(my_pipeline, X, y, cv=5, scoring='neg_mean_absolute_error')

    print("Average MAE score:", scores.mean())


def get_score(n_estimators, X, y):
    my_pipeline = Pipeline(steps=[
        ('preprocessor', SimpleImputer()),
        ('model', RandomForestRegressor(n_estimators, random_state=0))
    ])

    scores = -1 * cross_val_score(my_pipeline, X, y, cv=3, scoring='neg_mean_absolute_error')
    return scores.mean()

def find_the_best_score():
    # Read the data
    train_data = pd.read_csv('../../data/house_price/train.csv', index_col='Id')
    test_data = pd.read_csv('../../data/house_price/test.csv', index_col='Id')

    # Remove rows with missing target, separate target from predictors
    train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
    y = train_data.SalePrice
    train_data.drop(['SalePrice'], axis=1, inplace=True)

    # Select numerical col
    num_cols = [cname for cname in train_data.columns
                if train_data[cname].dtype in ['int64', 'float64']]
    X = train_data[num_cols].copy()
    X_test = test_data[num_cols].copy()

    result = {}
    for i in range(1, 9):
        result[50*i] = get_score(50*i, X, y)

    plt.plot(list(result.keys()), list(result.values()))
    plt.show()


if __name__ == '__main__':
    # cross_validation()
    find_the_best_score()