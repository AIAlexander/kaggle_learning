"""
Cross-validation: we run our modeling process on different subsets of the data
    to get multiple measures of model quality
For example, we could begin by dividing the data into 5 pieces, each 20% of the full dataset.
we say that we have broken the data into 5 folds.
In Experiment1, we use the first fold as a validation, set everything else as training data.
In Experiment2, we use the second fold as a validation, set everything else as training data.
And repeat this process.
"""

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score


def cross_validation():

    # Read the data
    data = pd.read_csv('../../data/melb_data.csv')

    # select subset of predictors
    cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
    X = data[cols_to_use]

    # Select target
    y = data.Price

    my_pipeline = Pipeline(steps=[
        ('preprocessor', SimpleImputer()),
        ('model', RandomForestRegressor(n_estimators=50, random_state=0))
    ])

    # Multiply by -1 since sklearn calculates *negative* MAE
    scores = -1 * cross_val_score(my_pipeline, X, y, cv=5, scoring='neg_mean_absolute_error')

    print("MAE scores:\n", scores)

    print("Average MAE score (across experiments):", scores.mean())


if __name__ == '__main__':
    cross_validation()