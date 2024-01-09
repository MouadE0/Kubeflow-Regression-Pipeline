import pandas as pa
import numpy as np
import json
import argparse
from pathlib import Path

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

def polynomial_regression(split_data: dict, score: float) -> float:

    prefix = 'gs://data-cs-kubeflow'
    
    # read data from args.preprocess

    train_data = pa.read_parquet(f"{prefix}/G5/train_data.parquet")
    test_data = pa.read_parquet(f"{prefix}/G5/test_data.parquet")

    input_col =  ["GR","RHOB","DTC","NEUT",'PE','DS_INDEX','ds_ref_id']
    output_col = ["DT_SHEAR"]
    # Check if the columns are in the data
    if not set(input_col).issubset(train_data.columns):
        raise ValueError(f"Training data is missing columns {set(input_col) - set(train_data.columns)}")
    
    x_train_small = train_data[input_col]
    y_train_small = train_data[output_col]

    x_test_small = test_data[input_col]
    y_test_small = test_data[output_col]


    # Add polynomial features
    poly = PolynomialFeatures(degree=2)
    x_train_poly = poly.fit_transform(x_train_small)
    x_test_poly = poly.fit_transform(x_test_small)

    y_train = np.array(y_train_small.values).reshape(y_train_small.shape[0])

    # Use Ridge Regression algorithm
    regr = Ridge(alpha=0.1)

    regr.fit(x_train_poly, y_train)

    # y_test = np.array(y_test_small.values).reshape(y_test_small.shape[0])

    # score = regr.score(x_test_poly,y_test)
    score = mean_squared_error(y_test_small, x_test_poly)

    print(f"The Score of Polynomial Regression is {score}")

    # Save output into file
    with open(args.score, 'w') as score_file:
        score_file.write(str(score))

    return score


# Three modules: docker f

if __name__ == '__main__':

    # Defining and parsing the command-line arguments
    parser = argparse.ArgumentParser(description='Polynomial Regression Modeling')
    parser.add_argument('--score', type=str)
    parser.add_argument('--split_data', type=dict)
    args = parser.parse_args()

    # Creating the directory where the output file will be created (the directory may or may not exist).
    Path(args.score).parent.mkdir(parents=True, exist_ok=True)
    
    polynomial_regression(split_data=args.split_data, score=args.score)
