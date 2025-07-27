# streamlit run app.py
import pickle
import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
import os
import sys
import sklearn
from sklearn.ensemble import RandomForestClassifier

# Display scikit-learn version for debugging
st.sidebar.info(f"scikit-learn version: {sklearn.__version__}")

# Configure the page
st.set_page_config(page_title="Diabetes Predictor", layout="wide")

# Link CSS File
try:
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("CSS file not found, using default styling")

@st.cache_resource
def load_model():
    """Load the model with caching for better performance"""
    try:
        # Try multiple possible paths for the model file
        possible_paths = [
            'Diabetes-Prediction-ML-Model.sav',
            os.path.join(os.path.dirname(__file__), 'Diabetes-Prediction-ML-Model.sav'),
            os.path.join(os.getcwd(), 'Diabetes-Prediction-ML-Model.sav')
        ]
        
        for model_path in possible_paths:
            if os.path.exists(model_path):
                try:
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    return model
                except Exception as pickle_error:
                    st.warning(f"Warning: Model loading issue detected. This might be due to scikit-learn version differences.")
                    st.warning(f"Error details: {str(pickle_error)}")
                    # Try alternative loading method
                    try:
                        import joblib
                        with open(model_path, 'rb') as f:
                            model = joblib.load(f)
                        return model
                    except Exception as joblib_error:
                        st.error(f"Failed to load model with both pickle and joblib: {str(joblib_error)}")
                        return None
        
        st.error("Model file 'Diabetes-Prediction-ML-Model.sav' not found. Please ensure the model file is in the same directory as this app.")
        return None
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Heading of the Web App
st.title('Diabetes Prediction Web App')
st.subheader("Powered by Machine Learning")
st.write('Easy Application For Diabetes Prediction Disease')

# Load model
Data = load_model()

if Data is None:
    st.stop()

# Input fields for user to enter data
Pregnancies = st.text_input('Pregnancies')
Glucose = st.text_input('Glucose')
BloodPressure = st.text_input('Blood Pressure')
SkinThickness = st.text_input('Skin Thickness')
Insulin = st.text_input('Insulin')
BMI = st.text_input('BMI')
DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function')
Age = st.text_input('Age')

# Create a DataFrame from user inputs
df = pd.DataFrame({
    'Pregnancies': [Pregnancies], 
    'Glucose': [Glucose], 
    'BloodPressure': [BloodPressure], 
    'SkinThickness': [SkinThickness], 
    'Insulin': [Insulin], 
    'BMI': [BMI], 
    'DiabetesPedigreeFunction': [DiabetesPedigreeFunction], 
    'Age': [Age]
}, index=[0])

# Run Model when 'Confirm' button is clicked
con = st.button('Confirm')
if con:
    try:
        # Check if all inputs are provided
        if not all([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]):
            st.error("Please fill in all fields before making a prediction.")
        else:
            # Convert input values to float for prediction
            df = df.astype(float)
            # Predict using the loaded model
            result = Data.predict(df)
            
            # Display the prediction result
            if result[0] == 0:
                st.success("✅ You do not have Diabetes.")
            else:
                st.error("⚠️ Unfortunately, you have diabetes. It is important to see a doctor as soon as possible.")
                
    except ValueError:
        st.error("Please enter valid numeric values for all fields.")
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")

# Footer
st.info('Developed By: [Ahmed Maher](https://www.linkedin.com/in/ahmed-maher-algohary)')
st.write('Click Here To Get In Touch �� : [LinkedIn](https://www.linkedin.com/in/ahmed-maher-algohary)')