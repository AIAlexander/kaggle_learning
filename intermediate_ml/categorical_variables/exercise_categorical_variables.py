import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder


def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)


def drop_column():

    # read the data
    X = pd.read_csv('../../data/house_price/train.csv', index_col='Id')
    X_test = pd.read_csv('../../data/house_price/test.csv', index_col='Id')

    # remove rows with missing target, separate target from predictors
    X.dropna(axis=0, subset=['SalePrice'], inplace=True)
    y = X.SalePrice
    X.drop(['SalePrice'], axis=1, inplace=True)

    # drop columns with missing value
    cols_with_missing = [col for col in X.columns if X[col].isnull().any()]
    X.drop(cols_with_missing, axis=1, inplace=True)
    X_test.drop(cols_with_missing, axis=1, inplace=True)

    # Break off validation set from training data
    X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

    low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and X_train_full[cname].dtype == 'object']
    num_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

    my_cols = low_cardinality_cols + num_cols
    X_train = X_train_full[my_cols].copy()
    X_valid = X_valid_full[my_cols].copy()

    s = X_train.dtypes == 'object'
    obj_cols = list(s[s].index)
    print(obj_cols)

    drop_X_train = X_train.select_dtypes(exclude=['object'])
    drop_X_valid = X_valid.select_dtypes(exclude=['object'])

    print("MAE Approach 1 (Drop column):")
    print(score_dataset(drop_X_train, drop_X_valid, y_train, y_valid))


def ordinal_encoding():
    # read the data
    X = pd.read_csv('../../data/house_price/train.csv', index_col='Id')
    X_test = pd.read_csv('../../data/house_price/test.csv', index_col='Id')

    # remove rows with missing target, separate target from predictors
    X.dropna(axis=0, subset=['SalePrice'], inplace=True)
    y = X.SalePrice
    X.drop(['SalePrice'], axis=1, inplace=True)

    # drop columns with missing value
    cols_with_missing = [col for col in X.columns if X[col].isnull().any()]
    X.drop(cols_with_missing, axis=1, inplace=True)
    X_test.drop(cols_with_missing, axis=1, inplace=True)

    # Break off validation set from training data
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

    # low_cardinality_cols = [cname for cname in X_train_full.columns if
    #                         X_train_full[cname].nunique() < 10 and X_train_full[cname].dtype == 'object']
    # num_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]
    #
    # my_cols = low_cardinality_cols + num_cols
    # X_train = X_train_full[my_cols].copy()
    # X_valid = X_valid_full[my_cols].copy()

    s = X_train.dtypes == 'object'
    obj_cols = list(s[s].index)
    print(obj_cols)

    # print the unique entries in both the training and validation data for the 'Condition2' column
    print("Unique value in 'Condition2' column in training data:", X_train['Condition2'].unique())
    print("\nUnique value in 'Condition2' column in validation data:", X_valid['Condition2'].unique())

    # label_X_train = X_train.copy()
    # label_X_valid = X_valid.copy()
    #
    # ordinal_encoder = OrdinalEncoder()
    # label_X_train[obj_cols] = ordinal_encoder.fit_transform(label_X_train[obj_cols])
    # label_X_valid[obj_cols] = ordinal_encoder.transform(label_X_valid[obj_cols])
    #
    # print("MAE Approach 2 (Ordinal Encoding):")
    # print(score_dataset(label_X_train, label_X_valid, y_train, y_valid))

    """
    There is a problem that 'RRAn' and 'RRNn' are not exist in the training data
    so need to write a custom ordinal encoder to deal with new categories 
    The simplest approach is to drop the problematic categorical columns
    """

    # Column that can be safely ordinal encoded
    good_label_cols = [col for col in obj_cols if set(X_valid[col]).issubset(set(X_train[col]))]
    print(len(good_label_cols))

    # Problematic columns that will be dropped from the dataset
    bad_label_cols = list(set(obj_cols) - set(good_label_cols))
    print('Categorical columns that will be ordinal encoded:', good_label_cols)
    print('\nCategorical columns that will be dropped from the dataset:', bad_label_cols)

    # Drop categorical columns that will not be encoded
    label_X_train = X_train.drop(bad_label_cols, axis=1)
    label_X_valid = X_valid.drop(bad_label_cols, axis=1)

    # Apply the ordinal encoder
    ordinal_encoder = OrdinalEncoder()
    label_X_train[good_label_cols] = ordinal_encoder.fit_transform(label_X_train[good_label_cols])
    label_X_valid[good_label_cols] = ordinal_encoder.transform(label_X_valid[good_label_cols])

    print("MAE Approach 2 (Ordinal Encoder):")
    print(score_dataset(label_X_train, label_X_valid, y_train, y_valid))


def one_hot_encoder():
    # read the data
    X = pd.read_csv('../../data/house_price/train.csv', index_col='Id')
    X_test = pd.read_csv('../../data/house_price/test.csv', index_col='Id')

    # remove rows with missing target, separate target from predictors
    X.dropna(axis=0, subset=['SalePrice'], inplace=True)
    y = X.SalePrice
    X.drop(['SalePrice'], axis=1, inplace=True)

    # drop columns with missing value
    cols_with_missing = [col for col in X.columns if X[col].isnull().any()]
    X.drop(cols_with_missing, axis=1, inplace=True)
    X_test.drop(cols_with_missing, axis=1, inplace=True)

    # Break off validation set from training data
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

    object_cols = [col for col in X_train.columns if X_train[col].dtype == 'object']

    # Investigating cardinality
    # Get the number of unique entries in each column with categorical data
    object_nunique = list(map(lambda col : X_train[col].nunique(), object_cols))
    d = dict(zip(object_cols, object_nunique))
    sorted(d.items(), key=lambda x : x[1])
    print(d.items())

    '''
    One-hot encoding can greatly expend the size of the dataset
    For this reason, we typically will only one-hot encode columns with low cardinality 
    and high cardinality columns can either be dropped from the dataset
    '''

    low_cardinality_cols = [col for col in object_cols if X_train[col].nunique() < 10]
    high_cardinality_cols = list(set(object_cols) - set(low_cardinality_cols))
    print("Categorical columns that will be one-hot encoded:", low_cardinality_cols)
    print("\nCategorical columns that will be dropped from the dataset:", high_cardinality_cols)

    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[low_cardinality_cols]))
    OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[low_cardinality_cols]))

    # change the index, put them back
    OH_cols_train.index = X_train.index
    OH_cols_valid.index = X_valid.index

    # training data remove the categorical columns
    num_X_train = X_train.drop(object_cols, axis=1)
    num_X_valid = X_valid.drop(object_cols, axis=1)

    # concat the categorical columns and the number columns
    OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
    OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

    print("MAE Approach 3(One-hot Encoder):")
    print(score_dataset(OH_X_train, OH_X_valid, y_train, y_valid))


if __name__ == '__main__':
    # drop_column()
    # ordinal_encoding()
    one_hot_encoder()