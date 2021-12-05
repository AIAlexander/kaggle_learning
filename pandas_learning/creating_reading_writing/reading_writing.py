import pandas as pd

# pandas can read the csv to a DataFrame, and a DataFrame can be written to a csv.


def reading_csv():
    data_frame = pd.read_csv('../../data/winemag-data-130k-v2.csv')
    print(data_frame.shape)
    print(data_frame.head())


def write_to_csv():
    data_frame = pd.DataFrame({'Apple': [80, 100], 'Banana': [40, 10]})
    data_frame.to_csv("../../data/test.csv", index=False, sep=',')


if __name__ == '__main__':
    # reading_csv()
    write_to_csv()