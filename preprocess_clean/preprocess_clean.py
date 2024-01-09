import pandas as pa
import numpy as np
import json
import argparse
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.impute import KNNImputer
from sklearn import preprocessing


def preprocess_clean(args) -> json:
    prefix = 'gs://data-cs-kubeflow'
    Path(f"{prefix}/G5").mkdir(parents=True, exist_ok=True)

    input_col =  ["GR","RHOB","DTC","NEUT",'PE','DS_INDEX', 'ds_ref_id']
    output_col = "DT_SHEAR"

    all_data = pa.read_parquet(f"{prefix}/all_data.parquet")

    # create the Labelencoder object
    le = preprocessing.LabelEncoder()
    # convert the categorical columns into numeric
    all_data['ds_ref_id'] = le.fit_transform(all_data['ds_ref_id'])
    # fill the missing values with mean 
    all_data.fillna(all_data.mean(), inplace=True)
    # save the data
    print("Save the train and test data into parquet files")
    all_data.to_parquet(f"{prefix}/G5/clean_data.parquet")
    print("clean_data.parquet saved in G5 folder")

    print("return json string of train and test data addresses")
    clean_data = json.dumps({
        'clean_data': f"{prefix}/G5/clean_data.parquet",
    })
    return clean_data


if __name__ == '__main__':

    # Defining and parsing the command-line arguments
    parser = argparse.ArgumentParser(description='Preprocess the data')
    args = parser.parse_args()

    # create the output directory if it doesn't exist
    # Path(args.score).parent.mkdir(parents=True, exist_ok=True)

    Path("/tmp/outputs/clean_data").mkdir(parents=True, exist_ok=True)
    clean_data = preprocess_clean(args)
    with open('/tmp/outputs/clean_data/data', 'w') as f:
        f.write(clean_data)
