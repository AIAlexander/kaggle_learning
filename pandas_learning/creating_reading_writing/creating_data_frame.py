import pandas as pd
# DataFrame is a table. Each entry corresponds to a row and a column

def create_data_frame():
    date_frame = pd.DataFrame({'Yes': [50, 21], 'No': [88, 98]})
    print(date_frame)


def create_string_data_frame():
    data_frame = pd.DataFrame({'Bob': ['hello', 'world'], 'Alice': ['I love', 'you']})
    print(data_frame)


def create_defined_index_data_frame():
    data_frame = pd.DataFrame({'Bob': ['hello', 'world'], 'Alice': ['I love', 'you']}, index=['Sentence1', 'Sentence2'])
    print(data_frame)

if __name__ == '__main__':
    # create_data_frame()
    # create_string_data_frame()
    create_defined_index_data_frame()
