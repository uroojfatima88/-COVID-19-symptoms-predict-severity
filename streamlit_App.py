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
# PREPROCESSING FUNCTION
# ---------------------------------------------------------
def preprocess_input(df):
    """Convert categorical/string columns to numeric"""
    df = df.copy()
    if 'gender' in df.columns:
        df['gender'] = df['gender'].map({0:0, 1:1, 'Female':0, 'Male':1})
    if 'vaccination_status' in df.columns:
        df['vaccination_status'] = df['vaccination_status'].map({0:0,1:1,'No':0,'Yes':1})
    # Add any other categorical mappings if needed
    return df

# ---------------------------------------------------------
# SAFE SCALING FUNCTION
# ---------------------------------------------------------
def safe_scale(data):
    """
    Accepts a list (single patient) or DataFrame (batch),
    returns scaled NumPy array, bypassing feature name checks.
    """
    if isinstance(data, list):
        df = pd.DataFrame([data], columns=FEATURE_COLUMNS)
        df = preprocess_input(df)
        return scaler.transform(df.to_numpy())
    elif isinstance(data, pd.DataFrame):
        df = data[FEATURE_COLUMNS].copy()
        df = preprocess_input(df)
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
        gender = st.selectbox("Gender", ['Female', 'Male'])
        vaccination_status = st.selectbox("Vaccinated", ['No', 'Yes'])
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
        cancer
