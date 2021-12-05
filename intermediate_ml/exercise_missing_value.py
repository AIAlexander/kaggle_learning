import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer


def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=10, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)


def preliminary_investigation():

    # read the data
    X_full = pd.read_csv('../data/house_price/train.csv', index_col='Id')
    X_test_full = pd.read_csv('../data/house_price/test.csv', index_col='Id')

    # Remove rows with missing target, separate target from predictors
    X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
    y = X_full.SalePrice
    X_full.drop(['SalePrice'], axis=1, inplace=True)

    X = X_full.select_dtypes(exclude=['object'])
    X_test = X_test_full.select_dtypes(exclude=['object'])

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

    print(X_train.head())

    # shape of training data
    print(X_train.shape)

    # Number of missing values in each column of training data
    missing_val_count_by_column = (X_train.isnull().sum())
    print(missing_val_count_by_column[missing_val_count_by_column > 0])


def drop_columns():
    # read the data
    X_full = pd.read_csv('../data/house_price/train.csv', index_col='Id')
    X_test_full = pd.read_csv('../data/house_price/test.csv', index_col='Id')

    # Remove rows with missing target, separate target from predictors
    X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
    y = X_full.SalePrice
    X_full.drop(['SalePrice'], axis=1, inplace=True)

    X = X_full.select_dtypes(exclude=['object'])
    X_test = X_test_full.select_dtypes(exclude=['object'])

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

    col_missing = [col for col in X_train.columns
                   if X_train[col].isnull().any()]

    reduced_X_train = X_train.drop(col_missing, axis=1)
    reduced_X_valid = X_valid.drop(col_missing, axis=1)

    print("MAE (Drop colums with missing value):")
    print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))


def simple_imputation():
    # read the data
    X_full = pd.read_csv('../data/house_price/train.csv', index_col='Id')
    X_test_full = pd.read_csv('../data/house_price/test.csv', index_col='Id')

    # Remove rows with missing target, separate target from predictors
    X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
    y = X_full.SalePrice
    X_full.drop(['SalePrice'], axis=1, inplace=True)

    X = X_full.select_dtypes(exclude=['object'])
    X_test = X_test_full.select_dtypes(exclude=['object'])

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

    my_imputation = SimpleImputer()
    imputed_X_train = pd.DataFrame(my_imputation.fit_transform(X_train))
    imputed_X_valid = pd.DataFrame(my_imputation.transform(X_valid))

    imputed_X_train.columns = X_train.columns
    imputed_X_valid.columns = X_valid.columns

    print("MAE (Imputation):")
    print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))

if __name__ == '__main__':
    drop_columns()
    simple_imputation()