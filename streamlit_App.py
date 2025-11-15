import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="COVID Severity Predictor", page_icon="🩺", layout="wide")

st.title("🩺 COVID-19 Severity Prediction App")


# ---------------------------------------------------------
# LOAD MODEL & SCALER
# ---------------------------------------------------------
MODEL_PATH = "best_covid_ann.h5"
SCALER_PATH = "scaler.save"

model, scaler = None, None

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    st.success("✅ Model & Scaler Loaded Successfully!")
except:
    st.error("❌ Could not load model or scaler.")
    st.stop()

# ---------------------------------------------------------
# FEATURES (Must match scaler)
# ---------------------------------------------------------
FEATURE_COLUMNS = [
    "age",
    "fever",
    "cough",
    "fatigue",
    "shortness_of_breath",
    "loss_of_smell",
    "headache",
    "diabetes",
    "hypertension",
    "heart_disease",
    "asthma",
    "cancer",
    "gender_Male",
    "gender_Other",
    "vaccination_status_Fully Vaccinated",
    "vaccination_status_Partially Vaccinated",
    "vaccination_status_Unvaccinated"
]

SEVERITY_MAP = {
    0: "🟢 Mild",
    1: "🟡 Moderate",
    2: "🔴 Severe"
}

# ---------------------------------------------------------
# ONE-HOT ENCODING FOR FORM INPUT
# ---------------------------------------------------------
def build_input_vector(values):
    age, gender, vax, *symptoms = values

    gender_Male = 1 if gender == "Male" else 0
    gender_Other = 1 if gender == "Other" else 0

    vax_fully = 1 if vax == "Fully Vaccinated" else 0
    vax_partial = 1 if vax == "Partially Vaccinated" else 0
    vax_unvax = 1 if vax == "Unvaccinated" else 0

    return [
        age,
        *symptoms,
        gender_Male,
        gender_Other,
        vax_fully,
        vax_partial,
        vax_unvax
    ]

# ---------------------------------------------------------
# PREDICTION FUNCTION
# ---------------------------------------------------------
def predict_severity(vector):
    X = np.array(vector).reshape(1, -1)
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)
    return np.argmax(pred)

# ---------------------------------------------------------
# SINGLE PATIENT FORM
# ---------------------------------------------------------
st.header("🔍 Predict Severity for One Patient")

if model is not None:

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", 1, 120, 30)
        gender = st.selectbox("Gender", ["Female", "Male", "Other"])
        vaccination_status = st.selectbox("Vaccination Status", [
            "Fully Vaccinated",
            "Partially Vaccinated",
            "Unvaccinated"
        ])
        fever = 1 if st.selectbox("Fever", ["No", "Yes"]) == "Yes" else 0
        cough = 1 if st.selectbox("Cough", ["No", "Yes"]) == "Yes" else 0

    with col2:
        fatigue = 1 if st.selectbox("Fatigue", ["No", "Yes"]) == "Yes" else 0
        sob = 1 if st.selectbox("Shortness of Breath", ["No", "Yes"]) == "Yes" else 0
        smell = 1 if st.selectbox("Loss of Smell", ["No", "Yes"]) == "Yes" else 0
        headache = 1 if st.selectbox("Headache", ["No", "Yes"]) == "Yes" else 0

    with col3:
        diabetes = 1 if st.selectbox("Diabetes", ["No", "Yes"]) == "Yes" else 0
        hypertension = 1 if st.selectbox("Hypertension", ["No", "Yes"]) == "Yes" else 0
        heart_disease = 1 if st.selectbox("Heart Disease", ["No", "Yes"]) == "Yes" else 0
        asthma = 1 if st.selectbox("Asthma", ["No", "Yes"]) == "Yes" else 0
        cancer = 1 if st.selectbox("Cancer", ["No", "Yes"]) == "Yes" else 0


if st.button("Predict Severity"):
    sample = build_input_vector([
        age, gender, vaccination,
        fever, cough, fatigue, sob,
        smell, headache,
        diabetes, hypertension, heart_disease, asthma, cancer
    ])

    severity = predict_severity(sample)
    st.success(f"### Patient Severity: {SEVERITY_MAP[severity]}")

# ---------------------------------------------------------
# BATCH CSV PREDICTION
# ---------------------------------------------------------
st.header("📊 Batch Prediction (CSV Upload)")

csv = st.file_uploader("Upload CSV", type=["csv"])

if csv:
    df = pd.read_csv(csv)

    # Create missing one-hot columns if needed
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0  # default

    df = df[FEATURE_COLUMNS]

    X_scaled = scaler.transform(df)
    preds = model.predict(X_scaled)
    df["predicted_class"] = np.argmax(preds, axis=1)
    df["severity"] = df["predicted_class"].map(SEVERITY_MAP)

    st.dataframe(df)

    st.download_button("⬇ Download Predictions", df.to_csv(index=False), "predictions.csv")


