import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc
)

from src.model_training import load_all_models
from src.preprocessing import load_scaler, load_columns


def load_data():

    df = pd.read_csv("data/Telco-Customer-Churn.csv")

    df = df.drop("Churn", axis=1)

    return df


def plot_roc_curve(model, X_test, y_test):

    prob = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, prob)

    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()

    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")

    ax.plot([0, 1], [0, 1], linestyle="--")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")

    ax.legend()

    return fig


def plot_confusion_matrix(model, X_test, y_test):

    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    return fig


