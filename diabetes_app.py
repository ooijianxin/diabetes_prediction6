# diabetes_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Set page configuration
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model and resources
@st.cache_resource
def load_resources():
    try:
        model = joblib.load('optimized_diabetes_model.pkl')
        scaler = joblib.load('scaler.pkl')
        feature_names = joblib.load('feature_names.pkl')
        return model, scaler, feature_names
    except:
        return None, None, None

model, scaler, feature_names = load_resources()

# App title and description
st.title("🩺 Diabetes Prediction App")
st.markdown("""
This app predicts the likelihood of diabetes based on health metrics using an optimized machine learning model.
**Disclaimer:** This is a predictive tool, not a medical diagnosis. Always consult healthcare professionals.
""")

# Show warning if model files aren't loaded
if model is None:
    st.error("""
    ❌ Model files not found. Please run the training scripts first:
    1. python data_preprocessing.py
    2. python model_selection.py  
    3. python model_optimization.py
    """)
    st.stop()

# Input form
st.header("Patient Information")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.slider("Pregnancies", 0, 20, 1)
    glucose = st.slider("Glucose (mg/dL)", 50, 300, 100)
    blood_pressure = st.slider("Blood Pressure (mm Hg)", 40, 140, 70)
    skin_thickness = st.slider("Skin Thickness (mm)", 5, 100, 20)

with col2:
    insulin = st.slider("Insulin (μU/mL)", 0, 300, 80)
    bmi = st.slider("BMI", 15.0, 50.0, 25.0, 0.1)
    dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5, 0.01)
    age = st.slider("Age (years)", 20, 100, 30)

# Calculate derived features
glucose_bmi = glucose * bmi / 1000
age_glucose = age * glucose / 1000
bp_bmi = blood_pressure * bmi / 1000
insulin_glucose = insulin * glucose / 10000

# Create input data
input_data = {
    'Pregnancies': pregnancies,
    'Glucose': glucose,
    'BloodPressure': blood_pressure,
    'SkinThickness': skin_thickness,
    'Insulin': insulin,
    'BMI': bmi,
    'DiabetesPedigreeFunction': dpf,
    'Age': age,
    'Glucose_BMI': glucose_bmi,
    'Age_Glucose': age_glucose,
    'BP_BMI': bp_bmi,
    'Insulin_Glucose': insulin_glucose
}

# Convert to DataFrame
input_df = pd.DataFrame([input_data])[feature_names]

# Prediction
if st.button("Predict Diabetes Risk", type="primary"):
    # Preprocess input
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)
    
    risk_percentage = prediction_proba[0][1] * 100
    
    # Display results
    st.subheader("Prediction Results")
    
    if risk_percentage < 30:
        risk_level = "Low Risk"
        color = "green"
        emoji = "✅"
    elif risk_percentage < 60:
        risk_level = "Moderate Risk"
        color = "orange"
        emoji = "⚠️"
    else:
        risk_level = "High Risk"
        color = "red"
        emoji = "❗"
    
    st.markdown(f"<h2 style='color:{color};'>{emoji} {risk_level}</h2>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='color:{color};'>Probability: {risk_percentage:.1f}%</h3>", unsafe_allow_html=True)
    
    # Progress bar
    st.progress(risk_percentage / 100)
    
    # Metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Probability of Diabetes", f"{risk_percentage:.1f}%")
    with col2:
        st.metric("Probability of No Diabetes", f"{100 - risk_percentage:.1f}%")
    
    # Recommendations
    st.subheader("Recommendations")
    
    if risk_percentage < 40:
        st.success("""
        **Maintenance Plan:**
        - Continue healthy lifestyle habits
        - Maintain regular physical activity
        - Schedule annual health check-ups
        """)
    elif risk_percentage < 70:
        st.warning("""
        **Prevention Plan:**
        - Consult with a healthcare provider
        - Increase physical activity
        - Focus on weight management
        - Reduce processed foods and sugars
        """)
    else:
        st.error("""
        **Action Plan:**
        - Schedule appointment with healthcare professional
        - Request comprehensive diabetes screening
        - Implement lifestyle changes immediately
        - Monitor health regularly
        """)

# About section
with st.expander("About This App"):
    st.markdown("""
    **Model Information:**
    - Algorithm: Optimized Random Forest
    - Training Data: Pima Indians Diabetes Dataset
    - Features: 12 health metrics including derived features
    
    **Normal Health Ranges:**
    - Glucose: <100 mg/dL (fasting)
    - BMI: 18.5-24.9
    - Blood Pressure: <120/80 mmHg
    
    **Disclaimer:** This tool provides statistical predictions, not medical diagnoses.
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Optimized Random Forest Model")