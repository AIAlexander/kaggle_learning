import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def histograms():
    filepath = '../../data/iris.csv'

    iris_data = pd.read_csv(filepath, index_col='Id')

    print(iris_data.head())

    # Use histograms show the petal length varies in the flower
    sns.distplot(a=iris_data['Petal Length (cm)'], kde=False)
    plt.show()


def density_plot():
    filepath = '../../data/iris.csv'

    iris_data = pd.read_csv(filepath, index_col='Id')

    sns.kdeplot(data=iris_data['Petal Length (cm)'], shade=True)
    plt.show()


def kde_plot_2D():
    filepath = '../../data/iris.csv'
    iris_data = pd.read_csv(filepath, index_col='Id')

    sns.jointplot(x=iris_data['Petal Length (cm)'], y=iris_data['Sepal Width (cm)'], kind='kde')
    plt.show()


def color_coded_plot():
    iris_set_path = '../../data/iris_setosa.csv'
    iris_ver_path = '../../data/iris_versicolor.csv'
    iris_vir_path = '../../data/iris_virginica.csv'

    iris_set_data = pd.read_csv(iris_set_path, index_col='Id')
    iris_ver_data = pd.read_csv(iris_ver_path, index_col='Id')
    iris_vir_data = pd.read_csv(iris_vir_path, index_col='Id')

    sns.distplot(a=iris_set_data['Petal Length (cm)'], label="Iris-setosa", kde=False)
    sns.distplot(a=iris_ver_data['Petal Length (cm)'], label="Iris-versicolor", kde=False)
    sns.distplot(a=iris_vir_data['Petal Length (cm)'], label="Iris-virginica", kde=False)

    plt.title("Histogram of Petal Lengths, by Species")

    # Force legend to appear
    plt.legend()

    plt.show()


def color_coded_kde():
    iris_set_path = '../../data/iris_setosa.csv'
    iris_ver_path = '../../data/iris_versicolor.csv'
    iris_vir_path = '../../data/iris_virginica.csv'

    iris_set_data = pd.read_csv(iris_set_path, index_col='Id')
    iris_ver_data = pd.read_csv(iris_ver_path, index_col='Id')
    iris_vir_data = pd.read_csv(iris_vir_path, index_col='Id')

    sns.kdeplot(data=iris_set_data['Petal Length (cm)'], label="Iris-setosa", shade=True)
    sns.kdeplot(data=iris_ver_data['Petal Length (cm)'], label="Iris-versicolor", shade=True)
    sns.kdeplot(data=iris_vir_data['Petal Length (cm)'], label="Iris-virginica", shade=True)

    plt.title("Distribution of Petal Lengths, by Species")

    # Force legend to appear
    plt.legend()

    plt.show()


if __name__ == '__main__':
    # histograms()
    # density_plot()
    # kde_plot_2D()
    # color_coded_plot()
    color_coded_kde()