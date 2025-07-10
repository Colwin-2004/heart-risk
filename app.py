import streamlit as st
import numpy as np
import pandas as pd
import pickle
import pytesseract
from PIL import Image
import os

st.set_page_config(page_title="Heart Risk Detector", page_icon="❤️")

# Load the model and the scaler
import joblib
model = joblib.load("heart_model_v2.pkl")
scaler = joblib.load("scaler_v2.pkl")  # Make sure this file exists



# Title and intro
st.title("💓 Heart Attack Risk Detector")
st.write("Upload a medical report image or manually fill the details below to estimate your heart attack risk.")

# OCR Function
def extract_info_from_image(image):
    text = pytesseract.image_to_string(image).lower()
    info = {}

    def find_value(keywords, cast_type=float):
        for line in text.split('\n'):
            for kw in keywords:
                if kw in line:
                    nums = [s for s in line.split() if s.replace('.', '', 1).isdigit()]
                    if nums:
                        try:
                            return cast_type(nums[0])
                        except:
                            pass
        return None

    info['age'] = find_value(['age'])
    info['sysBP'] = find_value(['systolic', 'sys bp', 'systolic bp', 'sysbp'])
    info['diaBP'] = find_value(['diastolic', 'dia bp', 'diastolic bp', 'diabp'])
    info['totChol'] = find_value(['cholesterol', 'total chol', 'totchol'])
    info['BMI'] = find_value(['bmi'])
    info['heartRate'] = find_value(['heart rate', 'pulse'])
    info['glucose'] = find_value(['glucose'])
    return info

# Image Upload and OCR
uploaded_image = st.file_uploader("📤 Upload your medical report image", type=["png", "jpg", "jpeg"])
auto_filled = {}

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="📄 Uploaded Report Preview", use_column_width=True)
    auto_filled = extract_info_from_image(image)
    st.success("✅ Information extracted from image where possible. Please review or fill the rest.")
else:
    auto_filled = {}

# Input helpers
def number_input(label, key, default, minv=0, maxv=300, step=1):
    val = auto_filled.get(key)
    try:
        val = float(val)
    except:
        val = default
    return st.number_input(label, min_value=float(minv), max_value=float(maxv), value=float(val), step=float(step))

def yes_no_input(label, key):
    val = auto_filled.get(key, None)
    default = 0 if val in [1, '1', 'yes'] else 1
    return st.radio(label, ['Yes', 'No'], index=default) == 'Yes'

# Numerical Inputs
st.subheader("📝 Enter or Confirm Your Health Details")

age = number_input("Age", "age", 50, 20, 100)
sysBP = number_input("Systolic BP (mm Hg)", "sysBP", 120, 80, 200)
diaBP = number_input("Diastolic BP (mm Hg)", "diaBP", 80, 50, 140)
totChol = number_input("Total Cholesterol (mg/dL)", "totChol", 200, 100, 400)
BMI = number_input("Body Mass Index", "BMI", 25.0, 10, 60)
heartRate = number_input("Heart Rate (bpm)", "heartRate", 70, 30, 180)
glucose = number_input("Glucose Level (mg/dL)", "glucose", 100, 50, 300)

# Binary Inputs
st.markdown("### 🧍 Personal & Lifestyle")

gender = st.radio("Gender", ["Male", "Female"])
gender_code = 1 if gender == "Male" else 0

smoker = yes_no_input("Do you smoke?", "currentSmoker")
cigsPerDay = number_input("Cigarettes per Day", "cigsPerDay", 0, 0, 60)

bpMeds = yes_no_input("On Blood Pressure Medication?", "BPMeds")
stroke = yes_no_input("History of Stroke?", "prevalentStroke")
hypertension = yes_no_input("Hypertension?", "prevalentHyp")
diabetes = yes_no_input("Diabetes?", "diabetes")

# Prediction
if st.button("🔍 Predict Risk"):
    input_df = pd.DataFrame([[
        gender_code,
        age,
        int(smoker),
        cigsPerDay,
        int(bpMeds),
        int(stroke),
        int(hypertension),
        int(diabetes),
        totChol,
        sysBP,
        diaBP,
        BMI,
        heartRate,
        glucose
    ]], columns=[
        'male', 'age', 'currentSmoker', 'cigsPerDay', 'BPMeds', 'prevalentStroke',
        'prevalentHyp', 'diabetes', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose'
    ])

    try:
        # Scale the input
        scaled_input = scaler.transform(input_df)

        # Predict
        prob = model.predict_proba(scaled_input)[0][1] * 100

        st.subheader(f"🩺 Estimated Heart Attack Risk: **{prob:.2f}%**")

        if prob >= 50:
            st.error("🔴 High risk. Please consult a cardiologist immediately.")
        elif prob >= 20:
            st.warning("🟠 Moderate risk. Consider lifestyle changes and regular check-ups.")
        else:
            st.success("🟢 Low risk. Keep up your healthy habits!")

        st.markdown("📌 **Summary of Health Concerns:**")
        if sysBP > 140 or diaBP > 90:
            st.write("- ⚠️ High Blood Pressure – Monitor and control regularly.")
        if totChol > 240:
            st.write("- ⚠️ High Cholesterol – Adjust diet and exercise.")
        if BMI > 30:
            st.write("- ⚠️ Obesity – Weight loss advised.")
        if smoker:
            st.write("- ⚠️ Smoking – Quitting can reduce heart risk.")
        if diabetes:
            st.write("- ⚠️ Diabetes – Manage blood sugar levels.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

