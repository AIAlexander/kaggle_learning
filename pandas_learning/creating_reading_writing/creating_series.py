import pandas as pd

# Series is a sequence of data values. Series is a list


def create_series():
    series = pd.Series([1, 2, 3, 4, 5])
    print(series)


def create_define_index_named_series():
    series = pd.Series([1, 2, 3], index=['Number1', 'Number2', 'Number3'], name="NumberList")
    print(series)


if __name__ == '__main__':
    # create_series()
    create_define_index_named_series()