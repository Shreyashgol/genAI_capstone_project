# Customer Churn Prediction AI Platform

An industry-grade machine learning application for predicting customer churn using advanced ML models with an attractive, modern UI.

## Features

- **Beautiful Modern UI**: Gradient backgrounds, smooth animations, and professional card-based layout
- **Multiple ML Models**: Choose from Logistic Regression, Decision Tree, or Random Forest
- **Feature Importance Analysis**: Understand which factors drive customer churn
- **Real-time Predictions**: Get instant churn predictions with probability scores
- **Interactive Visualizations**: ROC curves, confusion matrices, and feature importance plots
- **Comprehensive Input Forms**: Easy-to-use forms for entering customer data

## Pages

1. **Home**: Introduction and overview of the platform
2. **Models**: Select from three powerful ML models
3. **Model Detail**: 
   - Feature Importance visualization
   - Interactive prediction interface
   - Model performance metrics
4. **About**: Project information and technologies used

## Installation

1. Clone the repository or navigate to the project directory:
```bash
cd genAI_capstone_project
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

Start the Streamlit application:
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## Models Included

- **Logistic Regression**: Linear model ideal for interpretability
- **Decision Tree**: Tree-based model with clear decision rules
- **Random Forest**: Ensemble method for superior accuracy

## UI Highlights

- Gradient purple theme with smooth transitions
- Hover effects on cards and buttons
- Responsive layout that works on all screen sizes
- Professional color scheme and typography
- Interactive tabs for organized content

## Project Structure

```
genAI_capstone_project/
├── app.py                  # Main Streamlit application
├── data/                   # Dataset directory
├── models/                 # Trained ML models
├── src/                    # Source code modules
│   ├── preprocessing.py    # Data preprocessing
│   ├── model_training.py   # Model loading
│   └── evaluation.py       # Model evaluation
└── requirements.txt        # Python dependencies
```

## Making Predictions

1. Navigate to the Models page
2. Select your preferred model
3. Go to the "Make Prediction" tab
4. Fill in customer information
5. Click "Predict Churn" to get results

## Understanding Results

- **Green Result**: Low churn risk - customer likely to stay
- **Red Result**: High churn risk - consider retention strategies
- Probability scores show confidence level of prediction

## Contributing

Feel free to fork this project and submit pull requests for improvements!

## License

This project is open source and available for educational purposes.
