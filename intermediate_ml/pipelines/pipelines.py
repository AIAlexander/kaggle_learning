"""
Pipelines is a simple way to keep the data preprocessing and modeling code organized.

"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


def pipelines():
    data = pd.read_csv('../../data/melb_data.csv')

    y = data.Price
    X = data.drop(['Price'], axis=1)

    X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

    categorical_cols = [cname for cname in X_train_full.columns
                        if X_train_full[cname].nunique() < 10 and X_train_full[cname].dtype == 'object']
    num_cols = [cname for cname in X_train_full.columns
                if X_train_full[cname].dtype in ['int64', 'float64']]
    my_cols = categorical_cols + num_cols

    X_train = X_train_full[my_cols].copy()
    X_valid = X_valid_full[my_cols].copy()

    """
    Construct the full pipeline in three steps
    1. Define Preprocessing Steps.  
        imputes missing values in numerical data
        imputes missing values and applies a one-hot encoding to categorical data 
    """

    # Preprocessing for numerical data
    numerical_transformer = SimpleImputer(strategy='constant')

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, num_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    """
    2. Define the model
    define a random forest model 
    """

    model = RandomForestRegressor(n_estimators=100, random_state=0)

    """
    3. Create and Evaluate the Pipeline
    define a pipeline that bundles the preprocessing and modeling step
    """

    # Bunble preprocessing and modeling code in a pipeline
    my_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    # Preprocessing of training data, fit model
    my_pipeline.fit(X_train, y_train)

    # Preprocessing of validation data, get predictions
    preds = my_pipeline.predict(X_valid)

    # Evaluate the model
    score = mean_absolute_error(y_valid, preds)
    print('MAE:', score)

if __name__ == '__main__':
    pipelines()