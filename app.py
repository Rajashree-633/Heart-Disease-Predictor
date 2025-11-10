import streamlit as st
import pickle
import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt

# Load the dataset
df = pd.read_csv("HeartDiseaseTrain-Test.csv")
# Rename columns so they match standard names
df.rename(columns={
    'cholestoral': 'chol',
    'Max_heart_rate': 'thalach'
}, inplace=True)
# -------------------------
# LOAD MODEL AND SCALER
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "heart_disease_predictor.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

try:
    with open(model_path, "rb") as model_file:
        heart_disease_predictor = pickle.load(model_file)
    with open(scaler_path, "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    
except FileNotFoundError:
    st.error("⚠ Model or Scaler not found. Make sure both are in the same folder as app.py.")
    st.stop()


# -------------------------
# PAGE CONFIGURATION
# -------------------------
st.set_page_config(
    page_title="Heart Disease Predictor ❤",
    page_icon="❤",
    layout="centered"
)

# -------------------------
# THEME TOGGLE
# -------------------------
mode = st.sidebar.radio("🌓 Theme", ["Light Mode", "Dark Mode"])

if mode == "Dark Mode":
    bg_color = "#0D1117"
    text_color = "#E6EDF3"
    box_color = "rgba(36, 41, 46, 0.9)"
else:
    bg_color = "#E3F2FD"
    text_color = "#212121"
    box_color = "rgba(255, 255, 255, 0.9)"

# -------------------------
# CUSTOM CSS (Improved visibility)
# -------------------------
st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url('https://img.freepik.com/free-photo/abstract-blurred-hospital-clinic-interior_74190-5188.jpg');
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    background-color: {bg_color};
    color: {text_color};
}}

h1 {{
    text-align: center;
    color: #D32F2F;
    font-weight: 800;
}}

.form-box {{
    background: {box_color};
    backdrop-filter: blur(10px);
    padding: 30px;
    border-radius: 15px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.25);
    max-width: 500px;
    margin: auto;
    color: {text_color};
}}

/* --- Make input boxes readable --- */
input, select, textarea {{
    background-color: rgba(255, 255, 255, 0.9) !important;
    color: #000 !important;
    border: 1px solid #888 !important;
    border-radius: 8px !important;
    padding: 8px !important;
    font-size: 16px !important;
}}

/* --- Improve label visibility --- */
label {{
    font-weight: 600 !important;
    color: {text_color} !important;
}}

.result {{
    text-align: center;
    padding: 25px;
    border-radius: 15px;
    margin-top: 25px;
    font-size: 20px;
    font-weight: 600;
}}
.success {{
    background-color: rgba(76, 175, 80, 0.2);
    color: #2e7d32;
}}
.danger {{
    background-color: rgba(244, 67, 54, 0.2);
    color: #b71c1c;
}}
footer {{
    text-align: center;
    font-size: 14px;
    color: #555;
    margin-top: 50px;
}}
</style>
""", unsafe_allow_html=True)


# -------------------------
# HEADER
# -------------------------
st.markdown("<h1>💓 Heart Disease Risk Predictor</h1>", unsafe_allow_html=True)
st.write("<p style='text-align:center;'>An AI-based dashboard to predict your heart disease risk and visualize key health metrics.</p>", unsafe_allow_html=True)
st.write("---")

# -------------------------
# INPUT FORM
# -------------------------
st.markdown('<div class="form-box">', unsafe_allow_html=True)

age = st.number_input("Age", min_value=1, max_value=120, step=1)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200)
chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG Results (0–2)", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=10.0, step=0.1)
slope = st.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0–4)", [0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia (0=Normal, 1=Fixed Defect, 2=Reversible Defect)", [0, 1, 2])

# Input array
input_data = np.array([[age, 1 if sex == "Male" else 0, cp, trestbps, chol, fbs,
                        restecg, thalach, exang, oldpeak, slope, ca, thal]])

try:
    input_scaled = scaler.transform(input_data)
except Exception:
    input_scaled = input_data

# -------------------------
# PREDICTION
# -------------------------
if st.button("🔍 Predict My Risk"):
    prediction = heart_disease_predictor.predict(input_scaled)[0]
    if prediction == 1:
        st.markdown('<div class="result danger">⚠ <b>High Risk of Heart Disease Detected.</b><br>Please consult a doctor soon.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result success">✅ <b>Low Risk of Heart Disease.</b><br>Keep up your healthy lifestyle!</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------
# HEALTH CHART (Cholesterol vs Heart Rate)
# ---------------------------------
st.write("---")
st.subheader("🩺 Cholesterol vs Maximum Heart Rate")

# Ensure dataset is loaded (replace with your actual DataFrame variable name)
# Example: df = pd.read_csv("HeartDiseaseTrain-Test.csv")

fig, ax = plt.subplots(figsize=(6, 5))

# Plot all patients' data for comparison
ax.scatter(df['chol'], df['thalach'], color='lightgray', alpha=0.6, label='Population Data')

# Plot the user’s data point
ax.scatter(chol, thalach, color='red', s=150, label='Your Data')

# Label and grid styling
ax.set_xlabel("Cholesterol (mg/dl)")
ax.set_ylabel("Max Heart Rate")
ax.set_title("Your Health Indicator (Comparison View)")
ax.grid(True, linestyle="--", alpha=0.5)

# Auto-fit axis to include both dataset and user's data
x_min, x_max = df['chol'].min() - 10, df['chol'].max() + 10
y_min, y_max = df['thalach'].min() - 10, df['thalach'].max() + 10
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

# Add text annotation for user's point
ax.text(chol + 2,thalach+ 2, f"({chol}, {thalach})", fontsize=9, color='black')

# Add legend
ax.legend()

# Display the chart in Streamlit
st.pyplot(fig)

 
# INFO SECTION
# -------------------------
st.write("---")
st.subheader("ℹ About Heart Disease")
st.write("""
Heart disease is one of the leading causes of death worldwide. Early detection and lifestyle
changes can significantly reduce the risk. This tool uses AI to predict the likelihood of
heart disease based on medical parameters like cholesterol, blood pressure, and ECG readings.
""")

st.subheader(" Prevention Tips")
st.markdown("""
✅ Maintain a balanced diet (low in saturated fats).  
✅ Exercise for at least 30 minutes daily.  
✅ Avoid smoking and limit alcohol intake.  
✅ Get regular health checkups.  
✅ Manage stress through yoga or meditation.  
""")

# -------------------------
# FOOTER
# -------------------------
st.markdown("""
<footer>
Developed by <b>Rajashree Dasgupta</b> | Powered by Streamlit & Machine Learning  
</footer>
""", unsafe_allow_html=True)