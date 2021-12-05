import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer

def generate_test_prediction():
    # read the data
    X_full = pd.read_csv('../../data/house_price/train.csv', index_col='Id')
    X_test_full = pd.read_csv('../../data/house_price/test.csv', index_col='Id')

    # separate target from predictions
    X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
    y = X_full.SalePrice

    # remove rows with missing target
    X_full.drop(['SalePrice'], axis=1, inplace=True)

    # use only numerical predictors
    X = X_full.select_dtypes(exclude=['object'])
    X_test = X_test_full.select_dtypes(exclude=['object'])

    # break off validation set from training data
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

    # Preprocessed training and validation data
    # use SimpleImputer to put the median number for missing value
    my_imputer = SimpleImputer(strategy='median')

    final_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
    final_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

    final_X_train.columns = X_train.columns
    final_X_valid.columns = X_valid.columns

    # Define the model and fit model
    my_model = RandomForestRegressor(n_estimators=100, random_state=0)
    my_model.fit(final_X_train, y_train)

    # Get validation predictions and MAE
    preds_valid = my_model.predict(final_X_valid)
    print("MAE (My Approach):")
    print(mean_absolute_error(y_valid, preds_valid))

    # Part b: generate test predictions, using the same way
    fianl_X_test = pd.DataFrame(my_imputer.transform(X_test))
    fianl_X_test.columns = X_test.columns

    # get test predictions
    preds_test_valid = my_model.predict(fianl_X_test)

    # output to the csv
    output = pd.DataFrame({'Id': X_test.index, 'SalePrice': preds_test_valid})
    output.to_csv('../../data/house_price/submission.csv', index=False)

if __name__ == '__main__':
    generate_test_prediction()