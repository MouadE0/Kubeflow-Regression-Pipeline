import pandas as pa
import numpy as np
import json
import argparse
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.impute import KNNImputer
from sklearn import preprocessing


def preprocess_split(args) -> json:
    prefix = 'gs://data-cs-kubeflow'
    Path(f"{prefix}/G5").mkdir(parents=True, exist_ok=True)

    all_data = pa.read_parquet(f"{prefix}/G5/clean_data.parquet")

    # Split the data into train and test
    print("Split the data into train and test")
    train_data = all_data.sample(frac=0.8,random_state=0)
    test_data = all_data.drop(train_data.index)

    # save the data
    print("Save the train and test data into parquet files")
    train_data.to_parquet(f"{prefix}/G5/train_data.parquet")
    test_data.to_parquet(f"{prefix}/G5/test_data.parquet")
    print("train_data.parquet and test_data.parquet saved in G5 folder")

    print("return json string of train and test data addresses")
    split_data = json.dumps({
        'train_data': f"{prefix}/G5/train_data.parquet",
        'test_data': f"{prefix}/G5/test_data.parquet"
    })
    return split_data


if __name__ == '__main__':

    # Defining and parsing the command-line arguments
    parser = argparse.ArgumentParser(description='Preprocess the data')
    args = parser.parse_args()

    # create the output directory if it doesn't exist
    Path("/tmp/outputs/split_data").mkdir(parents=True, exist_ok=True)
    clean_data = preprocess_split(args)
    with open('/tmp/outputs/split_data/data', 'w') as f:
        f.write(clean_data)
