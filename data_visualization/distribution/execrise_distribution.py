import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def histogram():
    cancer_b_filepath = '../../data/cancer_b.csv'
    cancer_m_filepath = '../../data/cancer_m.csv'

    cancer_b_data = pd.read_csv(cancer_b_filepath, index_col="Id")
    cancer_m_data = pd.read_csv(cancer_m_filepath, index_col="Id")

    print(cancer_b_data.head())
    print(cancer_m_data.head())

    # Histogram to show the Area(mean) varies in the two cancer
    sns.distplot(a=cancer_b_data['Area (mean)'], kde=False)
    sns.distplot(a=cancer_m_data['Area (mean)'], kde=False)
    plt.show()


def two_d_kde_plot():
    cancer_b_filepath = '../../data/cancer_b.csv'
    cancer_m_filepath = '../../data/cancer_m.csv'

    cancer_b_data = pd.read_csv(cancer_b_filepath, index_col="Id")
    cancer_m_data = pd.read_csv(cancer_m_filepath, index_col="Id")

    print(cancer_b_data.head())
    print(cancer_m_data.head())

    sns.kdeplot(data=cancer_b_data['Radius (worst)'], shade=True)
    sns.kdeplot(data=cancer_m_data['Radius (worst)'], shade=True)
    plt.show()


if __name__ == '__main__':
    # histogram()
    two_d_kde_plot()