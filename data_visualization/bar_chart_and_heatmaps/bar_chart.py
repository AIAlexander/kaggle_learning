import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def bar_chart():
    flight_filepath = "../../data/flight_delays.csv"

    flight_data = pd.read_csv(flight_filepath, index_col="Month")

    # Set the width and
    plt.figure(figsize=(10, 6))

    # Add the title
    plt.title("Average Arrival Delay for Spirit Airlines Flights, by month")

    # Bar chart showing average arrival delay for Spirit Airlines flight by month
    sns.barplot(x=flight_data.index, y=flight_data['NK'])

    plt.ylabel("Arrival delay (in minutes)")

    plt.show()

if __name__ == '__main__':
    bar_chart()