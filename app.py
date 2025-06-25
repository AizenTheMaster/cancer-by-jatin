import streamlit as st
import numpy as np
import pickle

# 🔄 Load the model and encoders
with open("lung_cancer_model_full.sav", "rb") as file:
    model_data = pickle.load(file)

model = model_data["model"]
label_encoders = model_data["label_encoders"]

# 🧾 Extract dropdown classes
gender_classes = label_encoders['gender'].classes_
country_classes = label_encoders['country'].classes_
smoking_classes = label_encoders['smoking_status'].classes_
treatment_classes = label_encoders['treatment_type'].classes_

# 🖼️ Page Setup
st.set_page_config(page_title="🫁 Lung Cancer Predictor", layout="centered")
st.markdown("<h1 style='text-align:center;'>🫁 Lung Cancer Survival Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center; color:gray;'>Built with ❤️ by Jatin</h4>", unsafe_allow_html=True)
st.markdown("---")

# 🧾 Input Fields (Vertical Layout)
st.subheader("📋 Enter Patient Information:")

age = st.number_input("🎂 Age (years)", min_value=0)
duration = st.number_input("🕒 Treatment Duration (days)", min_value=0)
gender = st.selectbox("⚥ Gender", gender_classes)
country = st.selectbox("🌍 Country", country_classes)
smoking_status = st.selectbox("🚬 Smoking Status", smoking_classes)
cancer_stage = st.number_input("🎯 Cancer Stage (1–4)", min_value=1, max_value=4)
tumor_size = st.number_input("📏 Tumor Size (cm)", min_value=0.0, step=0.1)
treatment_type = st.selectbox("💊 Treatment Type", treatment_classes)
physical_activity = st.number_input("🏃 Physical Activity Level (0–10)", min_value=0)
quality_of_life = st.number_input("💫 Quality of Life Score (0–10)", min_value=0)
symptom_score = st.number_input("🤒 Symptom Severity Score (0–10)", min_value=0)
income_level = st.number_input("💸 Income Level", min_value=0)
family_history = st.selectbox("🧬 Family History of Cancer", ["Yes", "No"])

# 🔄 Encode categorical fields
gender_encoded = label_encoders["gender"].transform([gender])[0]
country_encoded = label_encoders["country"].transform([country])[0]
smoking_encoded = label_encoders["smoking_status"].transform([smoking_status])[0]
treatment_encoded = label_encoders["treatment_type"].transform([treatment_type])[0]
family_encoded = 1 if family_history == "Yes" else 0

# 🔢 Input array for model
input_data = np.array([[age, duration, gender_encoded, country_encoded,
                        smoking_encoded, cancer_stage, tumor_size,
                        treatment_encoded, physical_activity, quality_of_life,
                        symptom_score, income_level, family_encoded]])

# 🔍 Prediction
st.markdown("---")
if st.button("🔍 Predict Survival"):
    prediction = model.predict(input_data)[0]
    result_text = "🟢 Patient is Likely to Survive" if prediction == 1 else "🔴 High Risk: Patient May Not Survive"
    color = "green" if prediction == 1 else "red"
    
    st.markdown(f"<h3 style='text-align:center; color: {color}'>{result_text}</h3>", unsafe_allow_html=True)
