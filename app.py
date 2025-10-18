import streamlit as st
import pandas as pd
import joblib
from PIL import Image
from pathlib import Path

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="💓 Heart Disease Predictor", page_icon="❤️", layout="wide")

# -----------------------------
# Base Directory
# -----------------------------
BASE_DIR = Path(__file__).parent

# -----------------------------
# Custom CSS Styling
# -----------------------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #ffdde1 0%, #ee9ca7 100%);
    color: #222;
    font-family: 'Poppins', sans-serif;
}
.main-title {
    font-size: 3rem;
    color: #b30000;
    font-weight: 800;
    text-shadow: 2px 2px 8px rgba(0,0,0,0.2);
}
.sub-title {
    font-size: 1.2rem;
    color: #444;
    margin-bottom: 1.5rem;
}
.stButton button {
    background: linear-gradient(90deg, #ff4b2b, #ff416c);
    color: white;
    border: none;
    padding: 0.8rem 2rem;
    border-radius: 10px;
    font-weight: 700;
    transition: 0.3s ease;
    box-shadow: 0 4px 10px rgba(0,0,0,0.2);
}
.stButton button:hover {
    transform: scale(1.05);
    background: linear-gradient(90deg, #ff6f61, #ff9966);
}
.result-box {
    background-color: rgba(255,255,255,0.9);
    border-radius: 15px;
    padding: 1.5rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
}
.footer {
    text-align: center;
    font-size: 13px;
    color: #333;
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Model
# -----------------------------
MODEL_PATH = BASE_DIR / "heart_disease_model.pkl"
if not MODEL_PATH.exists():
    st.error("❌ Model file not found! Make sure 'heart_disease_model.pkl' is in the same directory.")
    st.stop()
model = joblib.load(MODEL_PATH)

# -----------------------------
# Images
# -----------------------------
heart_img_path =  heart_img_url = "https://cdn-icons-png.flaticon.com/512/2966/2966487.png"
predict_img_path = BASE_DIR / "heartpics.png"

# -----------------------------
# Header Section (Image + Title Side by Side)
# -----------------------------
col1, col2 = st.columns([1, 2])

with col1:
    if heart_img_path.exists():
        st.image(str(heart_img_path), width=280)
    else:
        st.warning("⚠️ 'heartpic.png' not found.")

with col2:
    st.markdown("<h1 class='main-title'>💓 Heart Disease Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Predict your heart health instantly using a trained Machine Learning model. Get accurate risk assessments and take control of your health today!</p>", unsafe_allow_html=True)

st.markdown("---")

# -----------------------------
# Input Form Section
# -----------------------------
st.markdown("### 🩺 Enter Your Medical Details")
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

# -----------------------------
# Prediction Logic
# -----------------------------
if submitted:
    input_data = pd.DataFrame([[
        age,
        1 if sex == 'Male' else 0,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal
    ]], columns=[
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
        'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
    ])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][prediction]

    st.markdown("---")
    st.markdown("### 💡 Prediction Result")
    result_container = st.container()

    with result_container:
        st.markdown("<div class='result-box'>", unsafe_allow_html=True)
        if predict_img_path.exists():
            st.image(str(predict_img_path), width=120)

        if prediction == 1:
            st.error(f"⚠️ **High Risk of Heart Disease Detected!**\n\n🧠 Confidence: {probability:.2%}")
        else:
            st.success(f"✅ **Low Risk. Heart Appears Healthy.**\n\n💪 Confidence: {probability:.2%}")
        st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# About Section
# -----------------------------
st.markdown("---")
st.markdown("""
### ℹ️ About This App  
This web app uses a **Machine Learning model** to predict the likelihood of heart disease based on user-provided medical data.  
It demonstrates **data preprocessing, model training, and deployment using Streamlit** — bringing AI-powered health insights directly to your screen.
""")

# -----------------------------
# Footer
# -----------------------------
st.markdown("<p class='footer'>👨‍💻 Developed by Sikander Ktk | Made with ❤️ using Streamlit & Scikit-learn</p>", unsafe_allow_html=True)


