import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error


def pipelines():

    # Read the data
    X_full = pd.read_csv('../../data/house_price/train.csv')
    X_test_full = pd.read_csv('../../data/house_price/test.csv')

    # remove rows with missing target and separate target from prediction
    X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
    y = X_full.SalePrice
    X_full.drop(['SalePrice'], axis=1, inplace=True)

    X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, train_size=0.8, test_size=0.2, random_state=0)

    categorical_cols = [cname for cname in X_train_full.columns
                        if X_train_full[cname].nunique() < 10 and X_train_full[cname].dtype == 'object']
    num_cols = [cname for cname in X_train_full.columns
                if X_train_full[cname].dtype in ['int64', 'float64']]

    my_cols = categorical_cols + num_cols
    X_train = X_train_full[my_cols].copy()
    X_valid = X_valid_full[my_cols].copy()
    X_test = X_test_full[my_cols].copy()

    # Preprocessing the numerical data
    numerical_transformer = SimpleImputer(strategy='median')

    # Preprocessing the categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, num_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    # Define model
    model = RandomForestRegressor(n_estimators=100, random_state=0)

    # Bundle preprocessing and modeling code in a pipeline
    my_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    my_pipeline.fit(X_train, y_train)

    preds = my_pipeline.predict(X_valid)
    print("MAE:")
    print(mean_absolute_error(y_valid, preds))

    # Generate test predictions
    preds_test = my_pipeline.predict(X_test)
    output = pd.DataFrame({
        'Id': X_test.index,
        'SalePrice': preds_test
    })
    output.to_csv('../../data/submission_pipeline.csv', index=False)

if __name__ == '__main__':
    pipelines()