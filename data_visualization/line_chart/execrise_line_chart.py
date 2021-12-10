import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def line_chart():
    # Load the data
    museum_filepath = "../../data/museum_visitors.csv"

    museum_data = pd.read_csv(museum_filepath, index_col='Date', parse_dates=True)

    plt.figure(figsize=(16, 6))
    sns.lineplot(data=museum_data)
    plt.title("Monthly Visitors to Los Angles City Museums")
    plt.show()

    # create the line chart to show the number of visitors to Avila Adobe
    plt.figure(figsize=(16, 6))
    sns.lineplot(data=museum_data['Avila Adobe'], label='Avila Adobe')
    plt.xlabel('Date')
    plt.show()


if __name__ == '__main__':
    line_chart()