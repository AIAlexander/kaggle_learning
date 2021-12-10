import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def heatmap():
    filepath = '../../data/ign_scores.csv'

    ign_data = pd.read_csv(filepath, index_col="Platform")

    plt.figure(figsize=(16, 6))

    sns.heatmap(data=ign_data, annot=True)

    plt.show()


if __name__ == '__main__':
    heatmap()