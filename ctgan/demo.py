"""Demo module."""

import pandas as pd

dataset_path = ['/Users/macharya/Downloads/CTGAN/preprocessed_dataset.csv','',''
dataset_path = '/spark/benchmark/mortgage/input/'

def load_demo():
    """Load the demo."""
    for file_name in dataset_path:
        
        return pd.read_csv(file_name)
