import joblib
import os

def load_all_models():
    base_path = "models"

    models = {}

    models["Logistic Regression"] = joblib.load(
        os.path.join(base_path, "logistic_regression_model.pkl")
    )

    models["Decision Tree"] = joblib.load(
        os.path.join(base_path, "decision_tree_model.pkl")
    )

    models["Random Forest"] = joblib.load(
        os.path.join(base_path, "random_forest_model.pkl")
    )

    return models