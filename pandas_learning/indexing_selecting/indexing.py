import pandas as pd
import pandas_learning.utils as ut


if __name__ == '__main__':
    data_frame = ut.read_wine_data()
    print(data_frame.head())