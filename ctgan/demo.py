"""Demo module."""

import pandas as pd

DEMO_URL = 'http://ctgan-demo.s3.amazonaws.com/census.csv.gz'


def load_demo():
    """Load the demo."""
    return pd.read_csv(DEMO_URL, compression='gzip')
