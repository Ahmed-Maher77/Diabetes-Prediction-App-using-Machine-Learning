# streamlit run streamlit.py
import pickle
import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
import os
import sys
import sklearn

# Configure the page
st.set_page_config(page_title="Diabetes Predictor", layout="wide")

# Link CSS File
try:
    with open ('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("CSS file not found, using default styling")

# Load the model from the saved file
try:
    # Try multiple possible paths for the model file
    possible_paths = [
        'Diabetes-Prediction-ML-Model.sav',
        os.path.join(os.path.dirname(__file__), 'Diabetes-Prediction-ML-Model.sav'),
        os.path.join(os.getcwd(), 'Diabetes-Prediction-ML-Model.sav')
    ]
    
    model_loaded = False
    for model_path in possible_paths:
        if os.path.exists(model_path):
            Data = pickle.load(open(model_path, 'rb'))
            model_loaded = True
            break
    
    if not model_loaded:
        st.error("Model file 'Diabetes-Prediction-ML-Model.sav' not found. Please ensure the model file is in the same directory as this app.")
        st.stop()
        
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Heading of the Web App
st.title('Diabetes Prediction Web App')
st.subheader("Powered by Machine Learning")
st.write('Easy Application For Diabetes Prediction Disease')

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
df = pd.DataFrame({'Pregnancies': [Pregnancies], 'Glucose': [Glucose], 'BloodPressure': [BloodPressure], 'SkinThickness': [SkinThickness], 
'Insulin': [Insulin], 'BMI': [BMI], 'DiabetesPedigreeFunction': [DiabetesPedigreeFunction], 'Age': [Age]}, index=[0])

# Run Model when 'Confirm' button is clicked
con = st.button('Confirm')
if con:
    # Convert input values to float for prediction
    df = df.astype(float)
    # Predict using the loaded model
    result = Data.predict(df)
    # Display the prediction result as a pop-up notification
    if result == 0:
        mycode = "<script>alert('You do not have Diabetes.')</script>"
        components.html(mycode, height=0, width=0)
    else:
        mycode = "<script>alert('Unfortunately, you have diabetes. It is important to see a doctor as soon as possible.')</script>"
        components.html(mycode, height=0, width=0)

# Footer
st.info('Developed By: [Ahmed Maher](https://www.linkedin.com/in/ahmed-maher-algohary)')
st.write('Click Here To Get In Touch 📬 : [LinkedIn](https://www.linkedin.com/in/ahmed-maher-algohary)')