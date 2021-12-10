import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_the_data():
    fifa_filepath = '../../data/fifa.csv'

    fifa_data = pd.read_csv(fifa_filepath, index_col='Date', parse_dates=True)

    # set the width and height of the figure
    plt.figure(figsize=(16, 6))

    # Line chart showing how FIFA rankings evolved over time
    sns.lineplot(data=fifa_data)

    plt.show()


if __name__ == '__main__':
    load_the_data()