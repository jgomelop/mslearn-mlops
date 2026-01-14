import argparse
import glob
import os
import pandas as pd
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def main(args):
    mlflow.autolog()
    
    # Log custom parameters
    mlflow.log_param("reg_rate", args.reg_rate)
    
    # read data
    df = get_csvs_df(args.training_data)
    
    # split data
    X_train, X_test, y_train, y_test = split_data(df)
    
    # train model
    model = train_model(args.reg_rate, X_train, X_test, y_train, y_test)


def get_csvs_df(path):
    if not os.path.exists(path):
        raise RuntimeError(f"Cannot use non-existent path provided: {path}")
    csv_files = glob.glob(f"{path}/*.csv")
    if not csv_files:
        raise RuntimeError(f"No CSV files found in provided data path: {path}")
    return pd.concat((pd.read_csv(f) for f in csv_files), sort=False)


def split_data(df):
    # From notebook train-classification-model.ipynb
    X, y = (
        df[
            [
                "Pregnancies",
                "PlasmaGlucose",
                "DiastolicBloodPressure",
                "TricepsThickness",
                "SerumInsulin",
                "BMI",
                "DiabetesPedigree",
                "Age",
            ]
        ].values,
        df["Diabetic"].values,
    )
    return train_test_split(X, y, test_size=0.30, random_state=0)


def train_model(reg_rate, X_train, X_test, y_train, y_test):
    # train model
    model = LogisticRegression(C=1 / reg_rate, solver="liblinear").fit(X_train, y_train)
    
    # Evaluate to trigger autolog metrics
    accuracy = model.score(X_test, y_test)
    
    return model


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--training_data", dest="training_data", type=str)
    parser.add_argument("--reg_rate", dest="reg_rate", type=float, default=0.01)

    # parse args
    args = parser.parse_args()

    # return args
    return args


# run script
if __name__ == "__main__":
    # parse args
    args = parse_args()

    # run main function
    main(args)
