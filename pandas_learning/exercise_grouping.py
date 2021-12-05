import pandas as pd


if __name__ == '__main__':
    reviews = pd.read_csv('../data/winemag-data-130k-v2.csv', index_col=0)
    pd.set_option("display.max_rows", 5)

    max_point = reviews.groupby('price').points.max().sort_index()
    print(max_point)

    variety_min_max = reviews.groupby('variety').price.agg(['min', 'max'])
    print(variety_min_max)

    sorted_varieties = variety_min_max.sort_values(by=['min', 'max'], ascending=False)
    print(sorted_varieties)

    reviewer_mean_ratings = reviews.groupby('taster_name').points.mean()
    print(reviewer_mean_ratings)

    country_variety_counts = reviews.groupby(['country', 'variety']).size().sort_values(ascending=False)
    print(country_variety_counts)