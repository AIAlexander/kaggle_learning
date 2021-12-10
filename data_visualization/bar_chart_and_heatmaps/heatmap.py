import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def heatmap():

    filepath = '../../data/flight_delays.csv'

    flight_data = pd.read_csv(filepath, index_col='Month')

    plt.figure(figsize=(14, 7))
    plt.title("Average Arrival Delay for Each Airline, by Month")
    sns.heatmap(data=flight_data, annot=True)
    plt.xlabel("Airline")
    plt.show()


if __name__ == '__main__':
    heatmap()