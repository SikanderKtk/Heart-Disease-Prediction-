import streamlit as st
import pandas as pd
import joblib

# Load the model
import os
model_path = os.path.join(os.path.dirname(__file__), 'heart_disease_model.pkl')
model = joblib.load(model_path)

st.title("💓 Heart Disease Prediction App")

st.write("Enter the following details to predict the risk of heart disease.")

# Input features
age = st.number_input("Age", min_value=1, max_value=120)
sex = st.selectbox("Sex", ['Male', 'Female'])
cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure")
chol = st.number_input("Serum Cholesterol (mg/dl)")
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG (0-2)", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved")
exang = st.selectbox("Exercise Induced Angina (0 = No, 1 = Yes)", [0, 1])
oldpeak = st.number_input("ST Depression Induced by Exercise")
slope = st.selectbox("Slope of ST Segment (0-2)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (0 = Normal, 1 = Fixed Defect, 2 = Reversible Defect)", [0, 1, 2])

# Predict button
if st.button("Predict"):
    input_data = pd.DataFrame([[age, 1 if sex == 'Male' else 0, cp, trestbps, chol, fbs, restecg,
                                 thalach, exang, oldpeak, slope, ca, thal]],
                               columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                                        'restecg', 'thalach', 'exang', 'oldpeak',
                                        'slope', 'ca', 'thal'])

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease!")
    else:
        st.success("✅ Low Risk. No Heart Disease Detected.")

