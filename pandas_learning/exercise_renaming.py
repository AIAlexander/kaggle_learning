import pandas as pd

if __name__ == '__main__':
    reviews = pd.read_csv('../data/winemag-data-130k-v2.csv', index_col=0)
    pd.set_option("display.max_rows", 5)

    renamed = reviews.rename(columns=dict(region_1='region', region_2='locale'))
    print(renamed)

    reindexed = reviews.rename_axis('wines', axis='rows')
    print(reindexed)
