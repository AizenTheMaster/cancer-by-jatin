# lung_cancer_app.py 💉 Lung Cancer Survival Prediction by Himanshu

import streamlit as st
import numpy as np
import pickle

# 🔄 Load model + encoders
with open("lung_cancer_model_full.sav", "rb") as file:
    model_data = pickle.load(file)

model = model_data["model"]
label_encoders = model_data["label_encoders"]

# 🧾 Extract encoder classes
gender_classes = label_encoders['gender'].classes_
country_classes = label_encoders['country'].classes_
smoking_classes = label_encoders['smoking_status'].classes_
treatment_classes = label_encoders['treatment_type'].classes_

# 🖼️ Page setup
st.set_page_config(page_title="Lung Cancer Survival Predictor", layout="centered")
st.title("💉 Lung Cancer Survival Predictor - Made by Jatin")
st.markdown("Enter patient information to predict the likelihood of survival.")

# 👉 Input Fields
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("🎂 Age", min_value=0)
    duration = st.number_input("🕒 Duration of Treatment (days)", min_value=0)
    gender = st.selectbox("⚥ Gender", gender_classes)
    country = st.selectbox("🌍 Country", country_classes)
    smoking_status = st.selectbox("🚬 Smoking Status", smoking_classes)
    symptom_score = st.number_input("🤒 Symptom Severity Score", min_value=0)

with col2:
    cancer_stage = st.number_input("🎯 Cancer Stage (1–4)", min_value=1, max_value=4)
    tumor_size = st.number_input("📏 Tumor Size (cm)", min_value=0.0, step=0.1)
    treatment_type = st.selectbox("💊 Treatment Type", treatment_classes)
    physical_activity = st.number_input("🏃 Physical Activity Level", min_value=0)
    quality_of_life = st.number_input("💫 Quality of Life Score", min_value=0)
    income_level = st.number_input("💸 Income Level", min_value=0)
    family_history = st.selectbox("🧬 Family History of Cancer", ["Yes", "No"])

# 🔄 Encode categorical fields
gender_encoded = label_encoders["gender"].transform([gender])[0]
country_encoded = label_encoders["country"].transform([country])[0]
smoking_encoded = label_encoders["smoking_status"].transform([smoking_status])[0]
treatment_encoded = label_encoders["treatment_type"].transform([treatment_type])[0]
family_encoded = 1 if family_history == "Yes" else 0

# 🧠 Final input array (13 values)
input_data = np.array([[age, duration, gender_encoded, country_encoded,
                        smoking_encoded, cancer_stage, tumor_size,
                        treatment_encoded, physical_activity, quality_of_life,
                        symptom_score, income_level, family_encoded]])

# 🔍 Predict
if st.button("🔍 Predict Survival"):
    prediction = model.predict(input_data)[0]
    result = "🟢 Survived" if prediction == 1 else "🔴 Did Not Survive"
    st.success(f"🎯 Prediction: {result}")
