import pandas as pd

if __name__ == '__main__':
    reviews = pd.read_csv('../data/winemag-data-130k-v2.csv', index_col=0)
    pd.set_option("display.max_rows", 5)

    print(reviews.points.dtypes)

    point_strings = reviews.points.astype('str')
    print(point_strings)

    missing_prices = reviews[pd.isnull(reviews.price)]
    print(len(missing_prices))

    reviews_per_region_with_nan = reviews.region_1.fillna('Unknown')
    reviews_per_region = reviews_per_region_with_nan.value_counts().sort_values(ascending=False)
    print(reviews_per_region)