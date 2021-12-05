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


def drop_columns_with_missing_value():
    """
    Directly drop the column
    Unless most values in the dropped columns are missing, the model loses a lot of information with the approach
    :return:
    """

    # load the data
    data = pd.read_csv('../data/melb_data.csv')

    # select the target
    y = data.Price

    # use only numerical predictors
    melb_predictors = data.drop(['Price'], axis=1)
    X = melb_predictors.select_dtypes(exclude=['object'])

    # divide data into training and validation subsets
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

    # get names of columns with missing values
    cols_with_missing = [col for col in X_train.columns
                         if X_train[col].isnull().any()]

    # drop columns in training and validation data
    reduced_X_train = X_train.drop(cols_with_missing, axis=1)
    reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)

    print("MAE from Approach 1 (Drop columns with missing value):")
    print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))


def simple_imputation():
    """
    use SimpleImputation to replace missing values with the mean value
    While statisticians have experimented with more complex ways to determine imputed

    :return:
    """


    # load the data
    data = pd.read_csv('../data/melb_data.csv')

    # select the target
    y = data.Price

    # use only numerical predictors
    melb_predictors = data.drop(['Price'], axis=1)
    X = melb_predictors.select_dtypes(exclude=['object'])

    # divide data into training and validation subsets
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

    # imputation
    my_imputation = SimpleImputer()
    imputed_X_train = pd.DataFrame(my_imputation.fit_transform(X_train))
    imputed_X_valid = pd.DataFrame(my_imputation.transform(X_valid))

    # Imputation removed column names; put them back
    imputed_X_train.columns = X_train.columns
    imputed_X_valid.columns = X_valid.columns

    print("MAE from Approach 2 (Simple Imputation):")
    print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))


def extension_to_imputation():
    """
    We impute the missing values for each column with missing entries in the original dataset,
    additionally, add a new column that shows the location of the imputed entries

    :return:
    """


    # load the data
    data = pd.read_csv('../data/melb_data.csv')

    # select the target
    y = data.Price

    # use only numerical predictors
    melb_predictors = data.drop(['Price'], axis=1)
    X = melb_predictors.select_dtypes(exclude=['object'])

    # divide data into training and validation subsets
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

    # get names of columns with missing values
    cols_with_missing = [col for col in X_train.columns
                         if X_train[col].isnull().any()]

    # make copy to avoid changing original data
    X_train_plus = X_train.copy()
    X_valid_plus = X_valid.copy()

    for col in cols_with_missing:
        X_train_plus[col + "_was_missing"] = X_train_plus[col].isnull()
        X_valid_plus[col + "_was_missing"] = X_valid_plus[col].isnull()

    # imputation
    my_imputation = SimpleImputer()
    imputed_X_train_plus = pd.DataFrame(my_imputation.fit_transform(X_train_plus))
    imputed_X_valid_plus = pd.DataFrame(my_imputation.transform(X_valid_plus))

    imputed_X_train_plus.columns = X_train_plus.columns
    imputed_X_valid_plus.columns = X_valid_plus.columns

    print("MAE from Approach 3(An Extension to Imputation):")
    print(score_dataset(imputed_X_train_plus, imputed_X_valid_plus, y_train, y_valid))



if __name__ == '__main__':
    drop_columns_with_missing_value()
    simple_imputation()
    extension_to_imputation()