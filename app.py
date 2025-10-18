import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="💓 Heart Disease Prediction App",
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# -------------------- HEADER SECTION --------------------
st.markdown("""
    <style>
        body {
            background-color: #f8f9fa;
        }
        .main-title {
            text-align: center;
            color: #E63946;
            font-size: 2.2rem;
            font-weight: 800;
        }
        .sub-title {
            text-align: center;
            color: #6c757d;
            font-size: 1.1rem;
            margin-bottom: 30px;
        }
        .footer {
            text-align: center;
            color: #888;
            margin-top: 40px;
            font-size: 0.9rem;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>💓 Heart Disease Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Check your heart health using a Machine Learning model</p>", unsafe_allow_html=True)

# Display an online banner image
st.image("https://www.istockphoto.com/photo/human-heart-with-blood-vessels-gm1266230179-371131556", use_container_width=True)

st.markdown("---")

# -------------------- LOAD MODEL --------------------
model = joblib.load("heart_disease_model.pkl")

# -------------------- USER INPUT FORM --------------------
st.sidebar.header("🩺 Enter Patient Information")

age = st.sidebar.number_input("Age", min_value=1, max_value=120)
sex = st.sidebar.selectbox("Sex", ['Male', 'Female'])
cp = st.sidebar.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.sidebar.number_input("Resting Blood Pressure")
chol = st.sidebar.number_input("Cholesterol (mg/dl)")
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.sidebar.selectbox("Resting ECG", [0, 1, 2])
thalach = st.sidebar.number_input("Max Heart Rate")
exang = st.sidebar.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.sidebar.number_input("ST Depression")
slope = st.sidebar.selectbox("Slope", [0, 1, 2])
ca = st.sidebar.selectbox("Major Vessels (0–3)", [0, 1, 2, 3])
thal = st.sidebar.selectbox("Thalassemia (0-Normal, 1-Fixed, 2-Reversible)", [0, 1, 2])

# -------------------- PREDICTION --------------------
if st.sidebar.button("🔍 Predict"):
    input_data = pd.DataFrame([[age, 1 if sex == 'Male' else 0, cp, trestbps, chol, fbs,
                                restecg, thalach, exang, oldpeak, slope, ca, thal]],
                              columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                                       'restecg', 'thalach', 'exang', 'oldpeak',
                                       'slope', 'ca', 'thal'])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][prediction]

    st.markdown("---")
    st.image("https://cdn-icons-png.flaticon.com/512/765/765613.png", width=120)

    if prediction == 1:
        st.error(f"⚠️ **High Risk of Heart Disease Detected!**\n\nConfidence: **{probability:.2%}**")
        st.image("https://cdn.pixabay.com/photo/2020/05/03/17/12/heart-5125174_1280.jpg", use_container_width=True)
    else:
        st.success(f"✅ **Low Risk — No Heart Disease Detected.**\n\nConfidence: **{probability:.2%}**")
        st.image("https://cdn.pixabay.com/photo/2017/01/06/19/15/heart-1957200_1280.jpg", use_container_width=True)

# -------------------- FOOTER --------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p class='footer'>Made with ❤️ using Streamlit | Powered by Machine Learning</p>", unsafe_allow_html=True)

