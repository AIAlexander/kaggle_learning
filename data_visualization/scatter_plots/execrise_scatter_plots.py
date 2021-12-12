import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def simple_scatter_plots():
    filepath = '../../data/candy.csv'

    candy_data = pd.read_csv(filepath, index_col='id')

    print(candy_data.head())

    # A scatter plot to show the relationship between 'sugarpercent' and 'winpercent'
    sns.scatterplot(x=candy_data['sugarpercent'], y=candy_data['winpercent'])
    # plt.show()

    # A scatter plot to show the relationship between 'sugarpercent' and 'winpercent' with a regression line
    sns.regplot(x=candy_data['sugarpercent'], y=candy_data['winpercent'])

    plt.show()


def color_coded_scatter_plot():
    filepath = '../../data/candy.csv'

    candy_data = pd.read_csv(filepath, index_col='id')

    print(candy_data.head())

    # A scatter plot to show the relationship between 'sugarpercent' and 'winpercent'. Use chocolate to color-code the points
    sns.scatterplot(x=candy_data['pricepercent'], y=candy_data['winpercent'], hue=candy_data['chocolate'])
    # plt.show()

    # A scatter plot to show the relationship between 'sugarpercent' and 'winpercent'. Use chocolate to color-code the points with two regression line
    sns.lmplot(x='pricepercent', y='winpercent', hue='chocolate', data=candy_data)
    # plt.show()

    plt.show()


def swam_scatter_plot():
    filepath = '../../data/candy.csv'

    candy_data = pd.read_csv(filepath, index_col='id')

    print(candy_data.head())

    sns.swarmplot(x='chocolate', y='winpercent', data=candy_data)
    plt.show()


if __name__ == '__main__':
    # simple_scatter_plots()
    # color_coded_scatter_plot()
    swam_scatter_plot()