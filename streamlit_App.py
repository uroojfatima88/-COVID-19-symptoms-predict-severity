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
st.write("Automatically loads your trained ANN model and scaler.")

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

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 1, 120, 30)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    vaccination = st.selectbox("Vaccination Status", [
        "Fully Vaccinated",
        "Partially Vaccinated",
        "Unvaccinated"
    ])

with col2:
    fever = st.selectbox("Fever", [0, 1])
    cough = st.selectbox("Cough", [0, 1])
    fatigue = st.selectbox("Fatigue", [0, 1])
    sob = st.selectbox("Shortness of Breath", [0, 1])

with col3:
    smell = st.selectbox("Loss of Smell", [0, 1])
    headache = st.selectbox("Headache", [0, 1])
    diabetes = st.selectbox("Diabetes", [0, 1])
    hypertension = st.selectbox("Hypertension", [0, 1])
    heart_disease = st.selectbox("Heart Disease", [0, 1])
    asthma = st.selectbox("Asthma", [0, 1])
    cancer = st.selectbox("Cancer", [0, 1])

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
