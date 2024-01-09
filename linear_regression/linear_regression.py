import pandas as pa
import numpy as np
import json
import argparse
from pathlib import Path
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

def linear_regression(split_data: dict, score: float) -> float:
    prefix = 'gs://data-cs-kubeflow'

    train_data = pa.read_parquet(f"{prefix}/G5/train_data.parquet")
    test_data = pa.read_parquet(f"{prefix}/G5/test_data.parquet")

    input_col =  ["GR","RHOB","DTC","NEUT",'PE','DS_INDEX','ds_ref_id']
    output_col = ["DT_SHEAR"]

    # define y_train_small and x_train_s
    x_train_small = train_data[input_col]
    y_train_small = train_data[output_col]

    x_test_small = test_data[input_col]
    y_test_small = test_data[output_col]

    y = np.array(y_train_small.values).reshape(y_train_small.shape[0])

    reg = LinearRegression().fit(x_train_small, y)
    score = mean_squared_error(y_test_small, reg.predict(x_test_small))
    
    # score = reg.score(x_test_small,y_test_small)
    print(f"The Score of Linear Regression is {score}")

    # Save output into file
    with open(args.score, 'w') as score_file:
        score_file.write(str(score))

    return score

if __name__ == '__main__':

    # Defining and parsing the command-line arguments
    parser = argparse.ArgumentParser(description='Linear Regression Modeling')
    parser.add_argument('--score', type=str)
    parser.add_argument('--split_data', type=dict)

    args = parser.parse_args()

    # Creating the directory where the output file will be created (the directory may or may not exist).
    Path(args.score).parent.mkdir(parents=True, exist_ok=True)

    linear_regression(split_data=args.split_data, score=args.score)
