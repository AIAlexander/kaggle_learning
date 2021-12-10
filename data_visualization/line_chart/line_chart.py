import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def line_chart():
    spotify_filepath = "../../data/spotify.csv"

    spotify_data = pd.read_csv(spotify_filepath, index_col="Date", parse_dates=True)

    # Set the width and height of the figure
    plt.figure(figsize=(14, 6))

    # Add title
    plt.title("Daily Global Streams of Popular Songs in 2017-2018")

    # Line chart showing daily global streams of each song
    sns.lineplot(data=spotify_data)

    plt.show()

    # Plot a subset of the data
    list(spotify_data.columns)

    """
    Plot a subset of the data
    """
    plt.figure(figsize=(14, 6))

    plt.title("Daily Global Streams of Popular Songs in 2017-2018")

    # Line chart showing daily global streams of 'Shape of You'
    sns.lineplot(data=spotify_data['Shape of You'], label="Shape of You")

    # Line chart showing daily global streams of 'Despacito'
    sns.lineplot(data=spotify_data['Despacito'], label="Despacito")

    # Add label for horizontal axis
    plt.xlabel("Date")

    plt.show()

if __name__ == '__main__':
    line_chart()