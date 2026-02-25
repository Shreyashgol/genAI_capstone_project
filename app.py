import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib
import os

from src.model_training import load_all_models
from src.preprocessing import preprocess_input, load_columns, load_scaler
from src.evaluation import (
    plot_roc_curve,
    plot_confusion_matrix,
)


# CONFIG
st.set_page_config(
    page_title="Customer Churn AI Dashboard",
    layout="wide"
)

BASE_PATH = "models"

# Load assets once (efficient)
models = load_all_models()
columns = load_columns()
scaler = load_scaler()


# SIDEBAR NAVIGATION
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "Model Comparison",
        "Prediction",
    ]
)


# HOME PAGE
if page == "Home":

    st.title("Customer Churn Prediction Dashboard")

    st.write("""
    This dashboard provides:

    • Model comparison  
    • ROC curves  
    • Confusion matrices   
    • Real-time prediction  
    """)


# MODEL COMPARISON PAGE
elif page == "Model Comparison":

    st.title("Model Comparison")

    model_name = st.selectbox(
        "Select Model",
        list(models.keys())
    )

    model = models[model_name]

    # Load test data
    df = pd.read_csv("data/Telco-Customer-Churn.csv")

    y = df["Churn"].map({"Yes": 1, "No": 0})

    X = df.drop("Churn", axis=1)

    X = pd.get_dummies(X)

    X = X.reindex(columns=columns, fill_value=0)

    # FIXED: Use actual scaler object
    X_scaled = scaler.transform(X)


    st.subheader("ROC Curve")

    roc_fig = plot_roc_curve(model, X_scaled, y)

    st.pyplot(roc_fig)


    st.subheader("Confusion Matrix")

    cm_fig = plot_confusion_matrix(model, X_scaled, y)

    st.pyplot(cm_fig)


# PREDICTION PAGE
elif page == "Prediction":

    st.title("Customer Prediction")

    model_name = st.selectbox(
        "Select Model",
        list(models.keys())
    )

    model = models[model_name]


    # USER INPUT UI
    SeniorCitizen = st.selectbox("Senior Citizen", ["No", "Yes"])

    tenure = st.slider("Tenure", 0, 72, 12)

    MonthlyCharges = st.number_input("Monthly Charges", 0.0, 200.0)

    TotalCharges = st.number_input("Total Charges", 0.0, 10000.0)


    # Feature engineering
    if tenure <= 12:
        TenureGroup = "Short"
    elif tenure <= 36:
        TenureGroup = "Medium"
    else:
        TenureGroup = "Long"

    ChargeRatio = TotalCharges / (tenure + 1)


    input_dict = {

        "SeniorCitizen": SeniorCitizen,
        "tenure": tenure,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges,
        "TenureGroup": TenureGroup,
        "ChargeRatio": ChargeRatio
    }


    if st.button("Predict"):

        processed = preprocess_input(input_dict)

        prediction = model.predict(processed)[0]

        prob = model.predict_proba(processed)[0][1]

        if prediction == 1:

            st.error(f"Customer WILL CHURN\nProbability: {prob:.2%}")

        else:

            st.success(f"Customer will NOT CHURN\nProbability: {(1-prob):.2%}")


