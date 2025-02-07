import pandas as pd

def load_data(filepath):
    """Loads the expense data from a CSV file."""
    return pd.read_csv(filepath)

def preprocess_data(data):
    """Preprocesses the data for use in models."""
    # Add necessary preprocessing steps here
    return data
 
