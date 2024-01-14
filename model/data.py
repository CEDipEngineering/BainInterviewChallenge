from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

# Placeholder paths, using pathlib for OS compatibility and robustness
root_path = Path(__file__).parent.parent.resolve()
_train_data_path = root_path / 'data' / 'train.csv'
_test_data_path =  root_path / 'data' / 'test.csv'

def load_data() -> tuple[np.ndarray, np.ndarray]:
    """
    Loads the train and test data into pandas DataFrames.
    Can be updated in the future, to access diferent datasources
    (if data is public, Pandas can actually just receive a URL to 
     S3 Bucket or Azure Storage Container, which is neat)
    """
    train = pd.read_csv(_train_data_path)
    test = pd.read_csv(_test_data_path)
    return train, test

if __name__ == "__main__":
    print(load_data()[0].loc[0])