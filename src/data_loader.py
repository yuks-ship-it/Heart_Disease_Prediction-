import pandas as pd

def load_data(path):
    """
    Load the dataset from a CSV file
    """
    return pd.read_csv(path)