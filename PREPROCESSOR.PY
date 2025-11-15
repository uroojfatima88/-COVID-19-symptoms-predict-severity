# preprocessor.py
# A clean reusable preprocessing module for the COVID dataset
# Works with both training scripts and Streamlit apps.

import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# -------------------------------------------------------------
# 1. Severity Label Creation (based on your project's rules)
# -------------------------------------------------------------
def create_severity_label(df):
    """
    Creates the severity label using rules:
      - 0 = Mild        (not hospitalized)
      - 1 = Moderate    (hospitalized but NOT ICU and alive)
      - 2 = Severe      (ICU or mortality)
    """

    required_cols = ['hospitalized', 'icu_admission', 'mortality']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' needed for creating severity")

    def assign(row):
        if int(row['mortality']) == 1:
            return 2
        if int(row['icu_admission']) == 1:
            return 2
        if int(row['hospitalized']) == 1:
            return 1
        return 0

    df['severity'] = df.apply(assign, axis=1)
    return df


# -------------------------------------------------------------
# 2. Cleaning & Dropping Unwanted Columns
# -------------------------------------------------------------
def clean_dataframe(df):
    """
    Removes columns that should not be part of the model.
    Keeps dataset clean and consistent for training.
    """

    drop_cols = ['hospitalized', 'icu_admission', 'mortality']

    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=col)

    return df


# -------------------------------------------------------------
# 3. Encode categorical columns automatically
# -------------------------------------------------------------
def encode_features(df):
    """
    Detects all object-type features and applies one-hot encoding.
    """
    cat_cols = [c for c in df.columns if df[c].dtype == 'object']
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df_encoded


# -------------------------------------------------------------
# 4. Scale numerical features using StandardScaler
# -------------------------------------------------------------
def scale_features(X_train, X_test, save_scaler=False, filename="scaler.save"):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if save_scaler:
        joblib.dump(scaler, filename)

    return X_train_scaled, X_test_scaled, scaler


# -------------------------------------------------------------
# 5. Full preprocess function (used in training + streamlit app)
# -------------------------------------------------------------
def preprocess_for_training(df, create_target=True):
    """
    Applies full preprocessing:
        - Create severity label
        - Drop unwanted columns
        - Separate features & target
        - One-hot encoding
    """

    if create_target:
        df = create_severity_label(df)

    df = clean_dataframe(df)

    X = df.drop(columns=['severity'])
    y = df['severity']

    X = encode_features(X)

    return X, y
