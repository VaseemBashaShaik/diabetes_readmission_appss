import streamlit as st
import joblib
import numpy as np
import pandas as pd
import json

st.title("Predict Patient Readmission Risk")

# Load model + preprocessor + info
model = joblib.load("models/gb_model.joblib")
pre = joblib.load("models/preprocessor.joblib")

with open("models/model_info.json", "r") as f:
    model_info = json.load(f)

STABLE_THRESHOLD = model_info.get("threshold", 0.45)

with st.form("prediction_form"):
    st.subheader("Enter Patient Details")

    race = st.selectbox("Race", ["Caucasian", "AfricanAmerican", "Asian", "Hispanic", "Other"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    age_group = st.selectbox("Age Group", ["Young", "Middle", "Elderly"])
    time_in_hospital = st.number_input("Time in Hospital (days)", 1, 30, 4)
    num_lab_procedures = st.number_input("Number of Lab Procedures", 1, 100, 40)
    num_medications = st.number_input("Number of Medications", 1, 80, 12)
    number_outpatient = st.number_input("Outpatient Visits", 0, 20, 0)
    number_emergency = st.number_input("Emergency Visits", 0, 10, 0)
    number_inpatient = st.number_input("Inpatient Visits", 0, 10, 1)
    admission_type_id = st.number_input("Admission Type ID", 1, 8, 1)
    discharge_disposition_id = st.number_input("Discharge Disposition ID", 1, 30, 1)
    admission_source_id = st.number_input("Admission Source ID", 1, 25, 7)

    submitted = st.form_submit_button("Predict Risk")

if submitted:
    example = {
        "race": race,
        "gender": gender,
        "age_group": age_group,
        "time_in_hospital": time_in_hospital,
        "num_lab_procedures": num_lab_procedures,
        "num_medications": num_medications,
        "number_outpatient": number_outpatient,
        "number_emergency": number_emergency,
        "number_inpatient": number_inpatient,
        "total_visits": number_outpatient + number_emergency + number_inpatient,
        "admission_type_id": admission_type_id,
        "discharge_disposition_id": discharge_disposition_id,
        "admission_source_id": admission_source_id
    }

    X_df = pd.DataFrame([example])
    for col in pre.feature_names_in_:
        if col not in X_df.columns:
            X_df[col] = np.nan
    X_df = X_df[pre.feature_names_in_]

    Xp = pre.transform(X_df)
    prob = model.predict_proba(Xp)[0, 1]
    pred = int(prob >= STABLE_THRESHOLD)

    risk_label = "High" if prob > 0.65 else "Medium" if prob > 0.4 else "Low"
    color = "red" if risk_label == "High" else "orange" if risk_label == "Medium" else "green"

    st.markdown(f"### 🧠 Predicted Readmission Probability: **{prob:.2f}**")
    st.markdown(f"<h3 style='color:{color}'>Risk Level: {risk_label}</h3>", unsafe_allow_html=True)
