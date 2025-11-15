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
# AUTO-LOAD MODEL & SCALER FROM FILES
# ---------------------------------------------------------
MODEL_PATH = "best_covid_ann.h5"
SCALER_PATH = "scaler.save"

model, scaler = None, None

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        st.success("✅ Model & Scaler Loaded Automatically!")
    except Exception as e:
        st.error(f"❌ Error loading model/scaler: {e}")
else:
    st.error("❌ Model or Scaler not found in directory. Make sure files exist:")
    st.code("best_covid_ann.h5\nscaler.save")

# ---------------------------------------------------------
# REQUIRED FEATURES — MUST MATCH TRAINING
# ---------------------------------------------------------
FEATURE_COLUMNS = [
    'age',
    'gender',
    'vaccination_status',
    'fever',
    'cough',
    'fatigue',
    'shortness_of_breath',
    'loss_of_smell',
    'headache',
    'diabetes',
    'hypertension',
    'heart_disease',
    'asthma',
    'cancer'
]

# ---------------------------------------------------------
# SEVERITY LABELS
# ---------------------------------------------------------
SEVERITY_MAP = {
    0: "🟢 Mild",
    1: "🟡 Moderate",
    2: "🔴 Severe"
}

# ---------------------------------------------------------
# SAFE SCALING FUNCTION
# ---------------------------------------------------------
def safe_scale(data):
    """
    Accepts a list (single patient) or DataFrame (batch),
    returns scaled NumPy array, bypassing feature name checks.
    """
    if isinstance(data, list):  # single patient
        df = pd.DataFrame([data], columns=FEATURE_COLUMNS)
        return scaler.transform(df.to_numpy())
    elif isinstance(data, pd.DataFrame):  # batch
        df = data[FEATURE_COLUMNS].copy()
        return scaler.transform(df.to_numpy())
    else:
        raise ValueError("Input must be list or DataFrame")

# ---------------------------------------------------------
# PREDICTION FUNCTION
# ---------------------------------------------------------
def predict_severity(data):
    if model is None or scaler is None:
        st.error("❌ Model or scaler not loaded.")
        return None

    data_scaled = safe_scale(data)
    pred = model.predict(data_scaled)
    severity_class = np.argmax(pred)
    return severity_class

# ---------------------------------------------------------
# SINGLE PATIENT FORM
# ---------------------------------------------------------
st.header("🔍 Predict Severity for One Patient")

if model is not None:

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", 1, 120, 30)
        gender = st.selectbox("Gender (0=Female, 1=Male)", [0, 1])
        vaccination_status = st.selectbox("Vaccinated", [0, 1])
        fever = st.selectbox("Fever", [0, 1])
        cough = st.selectbox("Cough", [0, 1])

    with col2:
        fatigue = st.selectbox("Fatigue", [0, 1])
        sob = st.selectbox("Shortness of Breath", [0, 1])
        smell = st.selectbox("Loss of Smell", [0, 1])
        headache = st.selectbox("Headache", [0, 1])

    with col3:
        diabetes = st.selectbox("Diabetes", [0, 1])
        hypertension = st.selectbox("Hypertension", [0, 1])
        heart_disease = st.selectbox("Heart Disease", [0, 1])
        asthma = st.selectbox("Asthma", [0, 1])
        cancer = st.selectbox("Cancer", [0, 1])

    if st.button("Predict Severity"):
        sample = [
            age, gender, vaccination_status, fever, cough,
            fatigue, sob, smell, headache,
            diabetes, hypertension, heart_disease, asthma, cancer
        ]

        severity = predict_severity(sample)

        if severity is not None:
            st.success(f"### Patient Severity: {SEVERITY_MAP[severity]}")

# ---------------------------------------------------------
# BATCH CSV PREDICTION
# ---------------------------------------------------------
st.header("📊 Batch Prediction (CSV Upload)")

csv = st.file_uploader("Upload CSV for prediction", type=["csv"])

if csv and model is not None:
    df = pd.read_csv(csv)

    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        st.error(f"Your CSV is missing columns: {missing}")
    else:
        # Safe scaling using wrapper
        X_scaled = safe_scale(df)
        preds = model.predict(X_scaled)
        df["predicted_class"] = np.argmax(preds, axis=1)
        df["severity"] = df["predicted_class"].map(SEVERITY_MAP)

        st.dataframe(df)

        csv_output = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download Predictions", csv_output, "predictions.csv", "text/csv")
