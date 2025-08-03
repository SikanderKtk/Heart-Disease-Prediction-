import streamlit as st
import pandas as pd
import joblib
from PIL import Image
import os

# Load model
model_path = os.path.join(os.path.dirname(__file__), 'heart_disease_model.pkl')
model = joblib.load(model_path)

# Load images
heart_img_path = os.path.join(os.path.dirname(__file__), "heartpic.png")
predict_img_path = os.path.join(os.path.dirname(__file__), "heartpics.png")
heart_img = Image.open(heart_img_path)
predict_img = Image.open(predict_img_path)

# Page styling
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

# Header with heart image
st.image(heart_img, width=150)
st.title("💓 Heart Disease Prediction App")
st.markdown(
    "<h4 style='text-align: center; color: grey;'>Check your heart health using machine learning</h4>",
    unsafe_allow_html=True,
)

st.markdown("---")

# Input form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120)
        sex = st.selectbox("Sex", ['Male', 'Female'])
        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
        trestbps = st.number_input("Resting Blood Pressure")
        chol = st.number_input("Cholesterol (mg/dl)")
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])

    with col2:
        restecg = st.selectbox("Resting ECG", [0, 1, 2])
        thalach = st.number_input("Max Heart Rate")
        exang = st.selectbox("Exercise Induced Angina", [0, 1])
        oldpeak = st.number_input("ST Depression")
        slope = st.selectbox("Slope", [0, 1, 2])
        ca = st.selectbox("Major Vessels (0–3)", [0, 1, 2, 3])
        thal = st.selectbox("Thalassemia (0-Normal, 1-Fixed, 2-Reversible)", [0, 1, 2])

    submitted = st.form_submit_button("🔍 Predict")

if submitted:
    input_data = pd.DataFrame([[age, 1 if sex == 'Male' else 0, cp, trestbps, chol, fbs, restecg,
                                thalach, exang, oldpeak, slope, ca, thal]],
                              columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                                       'restecg', 'thalach', 'exang', 'oldpeak',
                                       'slope', 'ca', 'thal'])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][prediction]

    # Show prediction image
    st.image(predict_img, width=120)

    # Display result
    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Disease!\nConfidence: {probability:.2%}")
    else:
        st.success(f"✅ Low Risk. No Heart Disease Detected.\nConfidence: {probability:.2%}")

# Optional footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; font-size: 13px;'>Made with ❤️ using Streamlit</p>",
    unsafe_allow_html=True,
)
