import pandas as pd

# Load the dataset from a CSV File
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Explore Our Dataset
def inspect_data(df):
    print("-"*30,"Dataset Shape","-"*30)
    print(df.shape)
    print("-"*30,"Columns Names","-"*30)
    print(df.columns)
    print("-"*30,"First 5 Rows","-"*30)
    print(df.head())
    print("-" * 30, "Dataset Information", "-" * 30)
    df.info()
    print("-" * 30, "Statistics", "-" * 30)
    print(df.describe())