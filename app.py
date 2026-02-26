import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from src.model_training import load_all_models
from src.preprocessing import preprocess_input, load_columns, load_scaler
from src.evaluation import plot_roc_curve, plot_confusion_matrix

# Page Configuration
st.set_page_config(
    page_title="Customer Churn Prediction AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #7e22ce 100%);
    }
    
    [data-testid="stSidebar"] {
        display: none;
    }
    
    .card {
        background: white;
        padding: 40px;
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        margin: 30px 0;
        transition: all 0.4s ease;
    }
    
    .card:hover {
        transform: translateY(-8px);
        box-shadow: 0 25px 70px rgba(0,0,0,0.4);
    }
    
    .model-card {
        background: linear-gradient(135deg, #ffffff 0%, #f0f4ff 100%);
        padding: 35px;
        border-radius: 20px;
        text-align: center;
        cursor: pointer;
        border: 4px solid transparent;
        transition: all 0.4s ease;
        min-height: 280px;
    }
    
    .model-card:hover {
        border-color: #7e22ce;
        transform: scale(1.08);
        box-shadow: 0 15px 40px rgba(126, 34, 206, 0.3);
    }
    
    .main-title {
        font-size: 4em;
        font-weight: 900;
        color: white;
        text-align: center;
        margin-bottom: 20px;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.4);
    }
    
    .subtitle {
        font-size: 1.4em;
        color: #e0e7ff;
        text-align: center;
        margin-bottom: 50px;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #7e22ce 0%, #ec4899 100%);
        color: white;
        border: none;
        padding: 18px 50px;
        font-size: 20px;
        font-weight: 700;
        border-radius: 50px;
        transition: all 0.4s ease;
    }
    
    .stButton>button:hover {
        transform: scale(1.08);
        box-shadow: 0 12px 30px rgba(126, 34, 206, 0.5);
    }
    
    .result-card {
        padding: 50px;
        border-radius: 25px;
        text-align: center;
        margin: 30px 0;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    }
    
    .section-header {
        color: white;
        font-size: 2.2em;
        font-weight: 700;
        text-align: center;
        margin: 40px 0 30px 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

# Load resources
@st.cache_resource
def load_resources():
    models = load_all_models()
    columns = load_columns()
    scaler = load_scaler()
    return models, columns, scaler

models, columns, scaler = load_resources()

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'intro'
if 'user_data' not in st.session_state:
    st.session_state.user_data = None
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None

# ==================== PAGE 1: INTRO ====================
if st.session_state.page == 'intro':
    st.markdown("<h1 class='main-title'>🚀 Customer Churn Prediction AI</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Predict customer churn with advanced machine learning algorithms</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown("""
        <div class='card'>
            <h2 style='color: #7e22ce; text-align: center; margin-bottom: 25px;'>Welcome to the Future of Customer Analytics</h2>
            <p style='font-size: 1.2em; line-height: 2; text-align: justify; color: #374151;'>
                Customer churn is one of the most critical challenges facing businesses today. 
                Our AI-powered platform leverages state-of-the-art machine learning algorithms to 
                predict which customers are likely to leave, enabling proactive retention strategies.
            </p>
            <p style='font-size: 1.2em; line-height: 2; text-align: justify; color: #374151; margin-top: 20px;'>
                With three powerful models - <strong style='color: #7e22ce;'>Logistic Regression</strong>, 
                <strong style='color: #7e22ce;'>Decision Tree</strong>, and <strong style='color: #7e22ce;'>Random Forest</strong> 
                - analyze customer behavior and make data-driven decisions.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<h2 class='section-header'>✨ Key Features</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='card'>
            <div style='font-size: 4em; margin-bottom: 20px;'>🎯</div>
            <h3 style='color: #7e22ce; margin-bottom: 15px;'>Multiple Models</h3>
            <p style='font-size: 1.1em; color: #6b7280; line-height: 1.8;'>
                Choose from three powerful ML algorithms with unique strengths.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='card'>
            <div style='font-size: 4em; margin-bottom: 20px;'>📊</div>
            <h3 style='color: #7e22ce; margin-bottom: 15px;'>Accurate Predictions</h3>
            <p style='font-size: 1.1em; color: #6b7280; line-height: 1.8;'>
                Get instant churn predictions with probability scores.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='card'>
            <div style='font-size: 4em; margin-bottom: 20px;'>⚡</div>
            <h3 style='color: #7e22ce; margin-bottom: 15px;'>Real-time Analysis</h3>
            <p style='font-size: 1.1em; color: #6b7280; line-height: 1.8;'>
                Instant results with detailed probability analysis.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🚀 Start Prediction", use_container_width=True, key="start_btn"):
            st.session_state.page = 'prediction'
            st.rerun()

# ==================== PAGE 2: PREDICTION INPUT ====================
elif st.session_state.page == 'prediction':
    if st.button("🏠 Back to Home", use_container_width=True):
            st.session_state.page = 'intro'
            st.session_state.user_data = None
            st.session_state.selected_model = None
            st.rerun()
    st.markdown("<h1 class='main-title'>📋 Enter Customer Information</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Fill in the details below to predict churn probability</p>", unsafe_allow_html=True)
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 👤 Customer Demographics")
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Has Partner", ["No", "Yes"])
            dependents = st.selectbox("Has Dependents", ["No", "Yes"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("### 📞 Phone Services")
            phone_service = st.selectbox("Phone Service", ["No", "Yes"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("### 🌐 Internet Services")
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 🛡️ Additional Services")
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
            st.markdown("</div>", unsafe_allow_html=True)
            
        
            st.markdown("### 💳 Billing Information")
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
            payment_method = st.selectbox("Payment Method", 
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
            monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 50.0, step=5.0)
            total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 500.0, step=50.0)
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            submitted = st.form_submit_button("🔮 Predict Churn", use_container_width=True)
        
        if submitted:
            st.session_state.user_data = {
                'gender': gender,
                'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
                'Partner': partner,
                'Dependents': dependents,
                'tenure': tenure,
                'PhoneService': phone_service,
                'MultipleLines': multiple_lines,
                'InternetService': internet_service,
                'OnlineSecurity': online_security,
                'OnlineBackup': online_backup,
                'DeviceProtection': device_protection,
                'TechSupport': tech_support,
                'StreamingTV': streaming_tv,
                'StreamingMovies': streaming_movies,
                'Contract': contract,
                'PaperlessBilling': paperless_billing,
                'PaymentMethod': payment_method,
                'MonthlyCharges': monthly_charges,
                'TotalCharges': total_charges
            }
            st.session_state.page = 'model_selection'
            st.rerun()

# ==================== PAGE 3: MODEL SELECTION ====================
elif st.session_state.page == 'model_selection':
    if st.button("🏠 Back to prediction", use_container_width=True):
            st.session_state.page = 'prediction'
            st.session_state.user_data = None
            st.session_state.selected_model = None
            st.rerun()
    st.markdown("<h1 class='main-title'>🤖 Select Your Model</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Choose a machine learning model to analyze the customer data</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='model-card'>
            <div style='font-size: 5em; margin-bottom: 20px;'>📊</div>
            <h2 style='color: #7e22ce; margin-bottom: 15px;'>Logistic Regression</h2>
            <p style='font-size: 1.1em; color: #6b7280; line-height: 1.8; margin-bottom: 20px;'>
                Linear model ideal for interpretability and baseline performance.
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Select Logistic Regression", use_container_width=True, key="lr_btn"):
            st.session_state.selected_model = 'Logistic Regression'
            st.session_state.page = 'result'
            st.rerun()
    
    with col2:
        st.markdown("""
        <div class='model-card'>
            <div style='font-size: 5em; margin-bottom: 20px;'>🌳</div>
            <h2 style='color: #7e22ce; margin-bottom: 15px;'>Decision Tree</h2>
            <p style='font-size: 1.1em; color: #6b7280; line-height: 1.8; margin-bottom: 20px;'>
                Tree-based model with clear decision rules and interactions.
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Select Decision Tree", use_container_width=True, key="dt_btn"):
            st.session_state.selected_model = 'Decision Tree'
            st.session_state.page = 'result'
            st.rerun()
    
    with col3:
        st.markdown("""
        <div class='model-card'>
            <div style='font-size: 5em; margin-bottom: 20px;'>🌲</div>
            <h2 style='color: #7e22ce; margin-bottom: 15px;'>Random Forest</h2>
            <p style='font-size: 1.1em; color: #6b7280; line-height: 1.8; margin-bottom: 20px;'>
                Ensemble method for superior accuracy and robustness.
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Select Random Forest", use_container_width=True, key="rf_btn"):
            st.session_state.selected_model = 'Random Forest'
            st.session_state.page = 'result'
            st.rerun()

# ==================== PAGE 4: RESULT ====================
elif st.session_state.page == 'result':
    model_name = st.session_state.selected_model
    model = models[model_name]
    
    model_icons = {
        'Logistic Regression': '📊',
        'Decision Tree': '🌳',
        'Random Forest': '🌲'
    }
    
    st.markdown(f"<h1 class='main-title'>{model_icons[model_name]} {model_name}</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Churn Prediction Results</p>", unsafe_allow_html=True)
    
    processed = preprocess_input(st.session_state.user_data)
    prediction = model.predict(processed)[0]
    prob = model.predict_proba(processed)[0]
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if prediction == 1:
            st.markdown(f"""
            <div class='result-card' style='background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); color: white;'>
                <div style='font-size: 5em; margin-bottom: 20px;'>⚠️</div>
                <h1 style='font-size: 3em; margin-bottom: 20px;'>HIGH CHURN RISK</h1>
                <div style='font-size: 6em; font-weight: 900; margin: 30px 0;'>{prob[1]:.1%}</div>
                <p style='font-size: 1.4em; margin-top: 20px;'>
                    This customer is likely to churn. Consider retention strategies!
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='result-card' style='background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white;'>
                <div style='font-size: 5em; margin-bottom: 20px;'>✅</div>
                <h1 style='font-size: 3em; margin-bottom: 20px;'>LOW CHURN RISK</h1>
                <div style='font-size: 6em; font-weight: 900; margin: 30px 0;'>{prob[0]:.1%}</div>
                <p style='font-size: 1.4em; margin-top: 20px;'>
                    This customer is likely to stay. Keep up the excellent service!
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<h2 class='section-header'>📊 Detailed Probability Analysis</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class='card' style='background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; text-align: center;'>
            <h2 style='font-size: 2.5em; margin-bottom: 15px;'>Will Stay</h2>
            <div style='font-size: 4em; font-weight: 900;'>{prob[0]:.1%}</div>
            <p style='font-size: 1.2em; margin-top: 15px;'>Retention Probability</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='card' style='background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); color: white; text-align: center;'>
            <h2 style='font-size: 2.5em; margin-bottom: 15px;'>Will Churn</h2>
            <div style='font-size: 4em; font-weight: 900;'>{prob[1]:.1%}</div>
            <p style='font-size: 1.2em; margin-top: 15px;'>Churn Probability</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Try Another Model", use_container_width=True):
            st.session_state.page = 'model_selection'
            st.rerun()
    
    with col2:
        if st.button("📝 New Prediction", use_container_width=True):
            st.session_state.page = 'prediction'
            st.session_state.user_data = None
            st.rerun()
    
    with col3:
        if st.button("🏠 Back to Home", use_container_width=True):
            st.session_state.page = 'intro'
            st.session_state.user_data = None
            st.session_state.selected_model = None
            st.rerun()
