# diabetes_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page configuration
st.set_page_config(
    page_title="Diabetes Risk Assessment",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the trained model and scaler
@st.cache_resource
def load_model():
    try:
        # Try to load the model files
        model = joblib.load('diabetes_model.pkl')
        scaler = joblib.load('scaler.pkl')
        
        # Define feature names explicitly to ensure order
        feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age',
                        'Glucose_BMI', 'Age_Glucose', 'BP_BMI', 'Insulin_Glucose']
        
        return model, scaler, feature_names
    except Exception as e:
        st.error(f"Error loading model files: {str(e)}")
        st.info("Please make sure diabetes_model.pkl and scaler.pkl are in the same directory.")
        return None, None, None

model, scaler, feature_names = load_model()

# App title and description
st.title("🩺 Diabetes Risk Assessment")
st.markdown("""
This tool assesses your risk of developing diabetes based on health metrics.
**Disclaimer:** This is a predictive tool, not a medical diagnosis. 
Always consult with healthcare professionals for medical advice.
""")

# Show warning if model files aren't loaded
if model is None:
    st.error("""
    ❌ Model files not found. Please make sure you have these files in the same directory:
    - diabetes_model.pkl
    - scaler.pkl
    """)
    
    # Add instructions for deployment
    with st.expander("Deployment Instructions"):
        st.markdown("""
        1. **For local testing**: Run the model training scripts first
        2. **For Streamlit Cloud**: Upload all files including .pkl files
        3. **File structure**:
           - diabetes_app.py
           - diabetes_model.pkl
           - scaler.pkl
           - requirements.txt
        """)
    st.stop()

# Create tabs for better organization
tab1, tab2 = st.tabs(["Health Assessment", "About Diabetes"])

with tab1:
    st.header("Health Information")
    
    # Create two columns for layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Basic Information")
        pregnancies = st.number_input("Number of Pregnancies", 0, 20, 0, 
                                    help="Number of times pregnant")
        age = st.number_input("Age (years)", 21, 100, 35, 
                            help="Age in years")
        
        st.subheader("Glucose Metabolism")
        glucose = st.number_input("Glucose Level (mg/dL)", 50, 300, 100,
                                help="Plasma glucose concentration after 2 hours in an oral glucose tolerance test")
        insulin = st.number_input("Insulin Level (μU/mL)", 0, 300, 80,
                                help="2-Hour serum insulin level")
    
    with col2:
        st.subheader("Body Measurements")
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", 40, 130, 70,
                                       help="Diastolic blood pressure")
        skin_thickness = st.number_input("Skin Thickness (mm)", 5, 100, 25,
                                       help="Triceps skin fold thickness")
        bmi = st.number_input("Body Mass Index (BMI)", 15.0, 50.0, 25.0, 0.1,
                            help="Body mass index (weight in kg/(height in m)^2)")
        
        st.subheader("Genetic Factor")
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5, 0.01,
                            help="A function that scores the likelihood of diabetes based on family history")
    
    # Calculate derived features (same as in training)
    glucose_bmi = glucose * bmi / 1000
    age_glucose = age * glucose / 1000
    bp_bmi = blood_pressure * bmi / 1000
    insulin_glucose = insulin * glucose / 10000
    
    # Create input data in the correct order
    input_data = [pregnancies, glucose, blood_pressure, skin_thickness,
                 insulin, bmi, dpf, age,
                 glucose_bmi, age_glucose, bp_bmi, insulin_glucose]
    
    # Convert to DataFrame with explicit column order
    input_df = pd.DataFrame([input_data], columns=feature_names)
    
    # Prediction button
    if st.button("Assess Diabetes Risk", type="primary", use_container_width=True):
        try:
            # Preprocess the input data
            input_scaled = scaler.transform(input_df)
            
            # Make prediction
            prediction = model.predict(input_scaled)
            prediction_proba = model.predict_proba(input_scaled)
            
            # Calculate risk percentage
            risk_percentage = prediction_proba[0][1] * 100
            
            # Display results with appropriate styling
            st.subheader("Risk Assessment")
            
            if risk_percentage < 25:
                risk_level = "Low Risk"
                color = "green"
                emoji = "✅"
            elif risk_percentage < 50:
                risk_level = "Moderate Risk"
                color = "orange"
                emoji = "⚠️"
            else:
                risk_level = "High Risk"
                color = "red"
                emoji = "❗"
            
            st.markdown(f"<h2 style='color:{color};'>{emoji} {risk_level}</h2>", 
                       unsafe_allow_html=True)
            st.markdown(f"<h3 style='color:{color};'>Probability: {risk_percentage:.1f}%</h3>", 
                       unsafe_allow_html=True)
            
            # Create a visual gauge
            st.progress(risk_percentage / 100)
            
            # Display detailed probability
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Probability of Diabetes", f"{risk_percentage:.1f}%")
            with col2:
                st.metric("Probability of No Diabetes", f"{100-risk_percentage:.1f}%")
            
            # Show key risk factors
            st.subheader("Key Risk Factors")
            
            risk_factors = []
            if glucose > 140:
                risk_factors.append(f"**High glucose level** ({glucose} mg/dL)")
            if bmi >= 30:
                risk_factors.append(f"**High BMI** ({bmi}) - Obese range")
            elif bmi >= 25:
                risk_factors.append(f"**Elevated BMI** ({bmi}) - Overweight range")
            if age > 45:
                risk_factors.append(f"**Age** ({age} years - increased risk category)")
            if dpf > 1.0:
                risk_factors.append(f"**Family history indication** (DPF: {dpf:.2f})")
            
            if risk_factors:
                for factor in risk_factors:
                    st.write(f"• {factor}")
            else:
                st.info("No major risk factors identified based on your inputs.")
            
            # Recommendations based on risk level
            st.subheader("Recommendations")
            
            if risk_percentage < 40:
                st.success("""
                **Maintenance Plan:**
                - Continue with your healthy lifestyle habits
                - Maintain regular physical activity
                - Eat a balanced diet
                - Schedule annual health check-ups
                """)
            elif risk_percentage < 70:
                st.warning("""
                **Prevention Plan:**
                - Consult with a healthcare provider
                - Increase physical activity
                - Focus on weight management if needed
                - Reduce processed foods and sugars
                """)
            else:
                st.error("""
                **Action Plan:**
                - **Schedule appointment with healthcare professional**
                - Request comprehensive diabetes screening
                - Implement lifestyle changes immediately
                - Monitor health regularly
                """)
            
        except Exception as e:
            st.error(f"An error occurred during assessment: {str(e)}")
            st.info("Please check that all input values are valid.")

with tab2:
    st.header("About Diabetes")
    
    st.markdown("""
    ### What is Diabetes?
    
    Diabetes is a chronic condition that affects how your body turns food into energy. 
    
    ### Key Risk Factors:
    
    - **High glucose levels** (≥126 mg/dL when fasting)
    - **Obesity or high BMI** (BMI ≥30 significantly increases risk)
    - **Family history** of diabetes
    - **High blood pressure** (≥140/90 mmHg)
    - **Age** (risk increases after 45 years)
    
    ### Normal Health Ranges:
    
    - **Glucose**: <100 mg/dL (fasting)
    - **BMI**: 18.5-24.9
    - **Blood Pressure**: <120/80 mmHg
    
    ### Prevention Strategies:
    
    1. **Maintain a healthy weight**
    2. **Exercise regularly**
    3. **Eat a balanced diet**
    4. **Get regular health check-ups**
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Model: Optimized Random Forest | **For educational purposes only**")

# Add a sidebar with additional information
with st.sidebar:
    st.header("About This Tool")
    st.markdown("""
    This diabetes risk assessment tool uses a machine learning model trained on clinical data.
    
    **How it works:**
    1. Enter your health information
    2. The model analyzes multiple health factors
    3. Get your personalized risk assessment
    4. Receive evidence-based recommendations
    
    **Note:** This tool provides statistical assessment, not medical diagnosis.
    """)
