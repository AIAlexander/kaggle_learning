import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def scatter_plots():
    insurance_filepath = "../../data/insurance.csv"

    insurance_data = pd.read_csv(insurance_filepath)

    print(insurance_data.head())

    # x:horizontal x-axis, y:vertical y-axis
    sns.scatterplot(x=insurance_data['bmi'], y=insurance_data['charges'])
    # plt.show()

    # add the regression line
    sns.regplot(x=insurance_data['bmi'], y=insurance_data['charges'])
    # plt.show()

    # color-coded scatter plots
    sns.scatterplot(x=insurance_data['bmi'], y=insurance_data['charges'],
                    hue=insurance_data['smoker'])
    # plt.show()

    # add two regression lines
    sns.lmplot(x="bmi", y="charges", hue="smoker", data=insurance_data)
    # plt.show()

    # swarm plot
    sns.swarmplot(x=insurance_data['smoker'], y=insurance_data['charges'])
    plt.show()


if __name__ == '__main__':
    scatter_plots()