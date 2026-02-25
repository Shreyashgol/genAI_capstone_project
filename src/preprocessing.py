import pandas as pd
import joblib
import os

BASE_PATH = "models"

def load_scaler():
    scaler_path = os.path.join(BASE_PATH, "scaler.pkl")
    scaler = joblib.load(scaler_path)

    return scaler


def load_columns():
    columns_path = os.path.join(BASE_PATH, "model_columns.pkl")
    columns = joblib.load(columns_path)

    return columns


def preprocess_input(input_dict):
    df = pd.DataFrame([input_dict])
    df = pd.get_dummies(df)
    training_columns = load_columns()
    df = df.reindex(columns=training_columns, fill_value=0)

    scaler = load_scaler()

    df_scaled = scaler.transform(df)

    return df_scaled