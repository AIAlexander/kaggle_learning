import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def line_chart():
    ign_filepath = "../../data/ign_scores.csv"

    ign_data = pd.read_csv(ign_filepath, index_col="Platform")

    # pc_row = ign_data.loc[ign_data.Platform == 'PC']
    # print(pc_row.max(axis=1))

    # Create a bar chart that shows the average score for racing games, for each platform
    plt.figure(figsize=(16, 4))
    sns.barplot(x=ign_data['Racing'], y=ign_data.index)
    plt.xlabel("")
    plt.title("Average score for racing games, by platform")
    plt.show()


if __name__ == '__main__':
    line_chart()