import pandas as pa
import numpy as np
import json
import argparse
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.impute import KNNImputer
from sklearn import preprocessing


def preprocess(args) -> json:
    prefix = 'gs://data-cs-kubeflow'
    Path(f"{prefix}/G5").mkdir(parents=True, exist_ok=True)


    all_data = pa.read_parquet(f"{prefix}/all_data.parquet")

    input_col =  ["GR","RHOB","DTC","NEUT",'PE','DS_INDEX', 'ds_ref_id']
    output_col = "DT_SHEAR"
    all_data = all_data[input_col + [output_col]]

    # create the Labelencoder object
    le = preprocessing.LabelEncoder()
    # convert the categorical columns into numeric
    all_data['ds_ref_id'] = le.fit_transform(all_data['ds_ref_id'])

    # ------------ Dealing with missing values ------------
    # fill the missing values using knnimputer
    imputer = KNNImputer(n_neighbors=5)

    # impute the missing values
    all_data[input_col] = imputer.fit_transform(all_data[input_col])

    # drop the missing values in the target column
    all_data[output_col] = all_data[output_col].dropna(axis=0, how='any')

    # ------------ Detect and drop outliers ------------
    # calculate the z-score for each data point
    z_scores = (all_data - all_data.mean()) / all_data.std()

    # remove the outliers with a z-score greater than 3
    threshold = 3
    all_data = all_data[(z_scores < threshold).all(axis=1)]

    # ------------ Split the data ------------
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
    split_data = preprocess(args)
    with open('/tmp/outputs/split_data/data', 'w') as f:
        f.write(split_data)
