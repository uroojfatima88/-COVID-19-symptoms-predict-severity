# streamlit_app.py
# Adorable, polished Streamlit app for training & using an ANN on your CSV dataset.
# Save file and run: streamlit run streamlit_app.py


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import joblib
import os
import tempfile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ---------------------------
# Page config & cute CSS
# ---------------------------
st.set_page_config(page_title="Adorable COVID ANN Playground", layout="centered", initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
    /* page background */
    .stApp { 
        background: linear-gradient(180deg, #fffaf0 0%, #f7fbff 100%);
        color: #333333;
    }
    .title { font-size:36px; font-weight:800; color:#2c2c54; }
    .subtitle { color:#6b6b83; margin-bottom: 1.5rem; }
    .card { background: #ffffff; padding: 16px; border-radius: 14px; box-shadow: 0 6px 30px rgba(44,44,84,0.06); }
    .small { font-size:13px; color:#6b6b83; }
    .stButton>button { background: linear-gradient(90deg,#ff7ab6,#7ad7ff); color: white; border: none; }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# Helper functions
# ---------------------------
@st.cache_data
def load_csv(file):
    return pd.read_csv(file)

@st.cache_resource
def build_ann(n_features, n_classes, hidden1=128, hidden2=64, dropout=0.3):
    model = Sequential()
    model.add(Dense(hidden1, activation='relu', input_shape=(n_features,)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    model.add(Dense(hidden2, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    if n_classes == 2:
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    else:
        model.add(Dense(n_classes, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def preprocess_df(df, target_col):
    """
    1) Drop rows with NA in target
    2) Separate X and y
    3) One-hot encode categorical features (drop_first=True)
    """
    df = df.copy()
    df = df.dropna(subset=[target_col])
    X = df.drop(columns=[target_col])
    y = df[target_col].reset_index(drop=True)
    # detect categorical columns (object or category)
    cat_cols = [c for c in X.columns if X[c].dtype == 'object' or X[c].dtype.name=='category']
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    return X, y

def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def format_report_text(report_str):
    """Nicely format sklearn classification_report text for display"""
    return report_str

def download_bytes(obj, filename, filetype='model'):
    """Helper: write object to a bytes buffer appropriate for download"""
    if filetype == 'model':
        with open(filename, 'rb') as f:
            return f.read()
    elif filetype == 'scaler':
        with open(filename, 'rb') as f:
            return f.read()

# ---------------------------
# Sidebar: upload & model settings
# ---------------------------
with st.sidebar:
    st.markdown("## 🧾 Data & Model")
    uploaded_file = st.file_uploader("Upload CSV file (your dataset)", type=["csv"])
    st.write("or use a demo dataset below")
    use_demo = st.checkbox("Use demo dataset (500 rows)", value=True)

    st.markdown("---")
    st.markdown("## ⚙️ Model hyperparameters")
    epochs = st.number_input("Epochs", min_value=5, max_value=500, value=50)
    batch_size = st.selectbox("Batch size", options=[16, 32, 64], index=1)
    hidden1 = st.number_input("Hidden layer 1 units", min_value=8, max_value=512, value=128)
    hidden2 = st.number_input("Hidden layer 2 units", min_value=8, max_value=512, value=64)
    dropout = st.slider("Dropout rate", min_value=0.0, max_value=0.6, value=0.3)
    st.markdown("---")
    st.markdown("You can also upload a pre-trained Keras model (.h5) and scaler (joblib).")
    model_file = st.file_uploader("Upload model (.h5)", type=["h5"])
    scaler_file = st.file_uploader("Upload scaler (.save/.pkl)", type=["save","pkl","joblib"])

# ---------------------------
# Main UI Header
# ---------------------------
st.markdown('<div class="title">🌸 COVID ANN Playground</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Train a friendly ANN on your CSV. Choose any target column. Visualize, download, predict.</div>', unsafe_allow_html=True)

# ---------------------------
# Load dataset (uploaded or demo)
# ---------------------------
if uploaded_file is not None:
    try:
        df = load_csv(uploaded_file)
        st.success("Loaded your CSV ✅")
    except Exception as e:
        st.error("Failed to load CSV: " + str(e))
        st.stop()
elif use_demo:
    # Create a small demo dataset with same-ish columns as your original file
    np.random.seed(42)
    n = 500
    df = pd.DataFrame({
        'age': np.random.randint(18, 90, n),
        'gender': np.random.choice(['Male','Female'], n),
        'vaccination_status': np.random.choice(['Unvaccinated','Partially Vaccinated','Fully Vaccinated'], n),
        'fever': np.random.randint(0,2,n),
        'cough': np.random.randint(0,2,n),
        'fatigue': np.random.randint(0,2,n),
        'shortness_of_breath': np.random.randint(0,2,n),
        'loss_of_smell': np.random.randint(0,2,n),
        'headache': np.random.randint(0,2,n),
        'diabetes': np.random.randint(0,2,n),
        'hypertension': np.random.randint(0,2,n),
        'heart_disease': np.random.randint(0,2,n),
        'asthma': np.random.randint(0,2,n),
        'cancer': np.random.randint(0,2,n),
        'hospitalized': np.random.randint(0,2,n),
        'icu_admission': np.random.randint(0,2,n),
        'mortality': np.random.randint(0,2,n),
    })
    st.info("Demo dataset loaded — use this to try the app before using your real data.")

else:
    st.info("Upload a CSV to begin or enable 'Use demo dataset' in the sidebar.")
    st.stop()

# Show a preview and basic info
with st.expander("Preview dataset (first 10 rows)"):
    st.dataframe(df.head(10))

st.markdown("### Choose target column (what do you want the model to predict?)")
target_col = st.selectbox("Target column", options=df.columns.tolist())

st.markdown("**Target value counts**")
st.write(df[target_col].value_counts().to_frame(name='count'))

# ---------------------------
# Preprocess & Train button
# ---------------------------
if st.button("✨ Preprocess & Train ANN"):
    # 1) Preprocess
    with st.spinner("Preprocessing dataset..."):
        try:
            X, y = preprocess_df(df, target_col)
        except Exception as e:
            st.error("Preprocess failed: " + str(e))
            st.stop()

        st.write(f"Features: {X.shape[1]} columns — Samples: {X.shape[0]}")
        st.write("Feature columns preview:")
        st.write(X.columns.tolist()[:30])

        # Determine if binary or multiclass
        unique_vals = pd.Series(y).dropna().unique()
        n_classes = len(unique_vals)
        st.write(f"Detected {n_classes} distinct target values.")

        # Convert target to suitable format
        if n_classes == 2:
            # binary: ensure values are 0/1
            # If labels aren't numeric, map them to integers
            if not pd.api.types.is_numeric_dtype(y):
                mapping = {val: i for i, val in enumerate(sorted(unique_vals))}
                y_mapped = y.map(mapping).astype(int)
            else:
                y_mapped = y.astype(int)
            y_for_model = y_mapped.values
            y_cat = y_for_model  # for train_test_split stratifiy
        else:
            # multiclass: factorize to integers, then one-hot
            y_int, uniques = pd.factorize(y)
            y_cat = to_categorical(y_int)
            y_for_model = y_int

    # 2) Train/test split
    test_size = 0.20
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_cat, test_size=test_size, random_state=42,
            stratify=(y_for_model if len(np.unique(y_for_model))>1 else None)
        )
    except Exception:
        # If stratify fails (e.g., single class), do a plain split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_cat, test_size=test_size, random_state=42
        )

    # 3) Scale features
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

    # 4) Class weights (helps with imbalance)
    y_integers = np.argmax(y_train, axis=1) if (len(y_train.shape) > 1) else y_train
    try:
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_integers), y=y_integers)
        class_weights_dict = {i: w for i, w in enumerate(class_weights)}
    except Exception:
        class_weights_dict = None

    st.write("Class weights (used for training):")
    st.write(class_weights_dict if class_weights_dict else "None")

    # 5) Build model
    n_features = X_train_scaled.shape[1]
    n_model_classes = 2 if (n_classes == 2) else y_cat.shape[1]
    st.write("Building model...")
    model = build_ann(n_features, n_model_classes, hidden1=hidden1, hidden2=hidden2, dropout=dropout)
    st.write(model.summary())

    # 6) Train model
    st.info("Training — this may take a little while depending on epochs & CPU/GPU...")
    ckpt_file = tempfile.NamedTemporaryFile(suffix=".h5", delete=False).name
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=0),
        ModelCheckpoint(ckpt_file, monitor='val_loss', save_best_only=True, verbose=0)
    ]

    history = model.fit(
        X_train_scaled, y_train,
        validation_split=0.15,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights_dict,
        callbacks=callbacks,
        verbose=0
    )
    st.success("Training complete! 🎉")

    # 7) Show training curves
    st.subheader("Training curves")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(history.history['accuracy'], label='train_acc')
    ax[0].plot(history.history.get('val_accuracy', []), label='val_acc')
    ax[0].set_title("Accuracy")
    ax[0].legend()
    ax[1].plot(history.history['loss'], label='train_loss')
    ax[1].plot(history.history.get('val_loss', []), label='val_loss')
    ax[1].set_title("Loss")
    ax[1].legend()
    st.pyplot(fig)

    # 8) Evaluate on test set
    st.subheader("Test evaluation")
    preds_proba = model.predict(X_test_scaled)
    if n_classes == 2:
        preds = (preds_proba > 0.5).astype(int).reshape(-1)
        truth = y_test.reshape(-1).astype(int)
    else:
        preds = np.argmax(preds_proba, axis=1)
        truth = np.argmax(y_test, axis=1)

    acc = accuracy_score(truth, preds)
    st.metric(label="Test Accuracy", value=f"{acc:.4f}")

    st.text("Classification report:")
    try:
        if n_classes == 2:
            report = classification_report(truth, preds)
            st.text(report)
        else:
            # try to reconstruct class names (if original target was categorical)
            _, uniq_names = pd.factorize(df[target_col])
            if len(uniq_names) == len(np.unique(truth)):
                report = classification_report(truth, preds, target_names=uniq_names)
            else:
                report = classification_report(truth, preds)
            st.text(report)
    except Exception as e:
        st.write("Could not produce classification report:", e)

    # Confusion matrix
    cm = confusion_matrix(truth, preds)
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    st.pyplot(fig2)

    # 9) Save model & scaler to download
    model_save_path = "adorable_ann_model.h5"
    scaler_save_path = "adorable_scaler.save"
    model.save(model_save_path)
    joblib.dump(scaler, scaler_save_path)

    with open(model_save_path, "rb") as f:
        model_bytes = f.read()
    with open(scaler_save_path, "rb") as f:
        scaler_bytes = f.read()

    st.download_button("Download trained model (.h5)", data=model_bytes, file_name="adorable_ann_model.h5")
    st.download_button("Download scaler (joblib)", data=scaler_bytes, file_name="adorable_scaler.save")

    # Store model & scaler in session for predictions
    st.session_state['model'] = model
    st.session_state['scaler'] = scaler
    st.session_state['feature_columns'] = X.columns.tolist()
    st.session_state['target_info'] = {
        'n_classes': n_classes,
        'unique_values': list(unique_vals) if n_classes <= 20 else []
    }

# ---------------------------
# If user uploaded model & scaler, load into session
# ---------------------------
if model_file is not None and scaler_file is not None:
    try:
        tmp_model = tempfile.NamedTemporaryFile(suffix=".h5", delete=False).name
        with open(tmp_model, "wb") as f:
            f.write(model_file.read())
        tmp_scaler = tempfile.NamedTemporaryFile(suffix=".save", delete=False).name
        with open(tmp_scaler, "wb") as f:
            f.write(scaler_file.read())
        loaded_model = load_model(tmp_model)
        loaded_scaler = joblib.load(tmp_scaler)
        st.session_state['model'] = loaded_model
        st.session_state['scaler'] = loaded_scaler
        st.success("Uploaded model & scaler loaded into app session.")
    except Exception as e:
        st.error("Failed to load uploaded model/scaler: " + str(e))

# ---------------------------
# Single prediction UI
# ---------------------------
st.markdown("---")
st.markdown("## 🔮 Make a single prediction")

if 'model' in st.session_state and 'scaler' in st.session_state and 'feature_columns' in st.session_state:
    cols = st.session_state['feature_columns']
    with st.form("single-pred-form"):
        st.write("Fill input values for features (or leave zeros/defaults).")
        user_input = {}
        for c in cols:
            if 'age' in c.lower():
                user_input[c] = st.number_input(c, value=30)
            elif any(k in c.lower() for k in ['gender','vaccination']):
                user_input[c] = st.text_input(c, value="")
            else:
                # binary features fallback
                user_input[c] = st.number_input(c, min_value=0, max_value=1, value=0)
        submitted = st.form_submit_button("Predict")
    if submitted:
        row = pd.DataFrame([user_input])
        # align columns
        for col in cols:
            if col not in row.columns:
                row[col] = 0
        row = row[cols]
        row_scaled = st.session_state['scaler'].transform(row.values)
        proba = st.session_state['model'].predict(row_scaled)
        if proba.shape[1] == 1:
            pred = int((proba > 0.5).astype(int)[0][0])
            st.write(f"Predicted label: **{pred}** — Probability: **{float(proba[0][0]):.4f}**")
        else:
            pred = int(np.argmax(proba, axis=1)[0])
            st.write(f"Predicted class index: **{pred}**")
            st.write("Probabilities (per class):")
            st.write([float(x) for x in proba[0]])
else:
    st.info("Train a model in the app or upload a pre-trained model + scaler to use prediction features.")

# ---------------------------
# Batch prediction UI
# ---------------------------
st.markdown("---")
st.markdown("## 📂 Batch prediction (upload CSV)")
batch_file = st.file_uploader("Upload CSV for batch prediction (must have same feature columns)", key="batch_pred")

if batch_file is not None:
    if 'model' not in st.session_state or 'scaler' not in st.session_state or 'feature_columns' not in st.session_state:
        st.warning("Please train a model in this app or upload a model & scaler before using batch prediction.")
    else:
        batch_df = pd.read_csv(batch_file)
        st.write("Preview uploaded batch:")
        st.dataframe(batch_df.head())
        # preprocess similarly: one-hot encode categorical columns to match training columns
        batch_X = batch_df.copy()
        cat_cols = [c for c in batch_X.columns if batch_X[c].dtype == 'object' or batch_X[c].dtype.name=='category']
        if cat_cols:
            batch_X = pd.get_dummies(batch_X, columns=cat_cols, drop_first=True)
        # align training columns
        for c in st.session_state['feature_columns']:
            if c not in batch_X.columns:
                batch_X[c] = 0
        batch_X = batch_X[st.session_state['feature_columns']]
        batch_scaled = st.session_state['scaler'].transform(batch_X.values)
        probs = st.session_state['model'].predict(batch_scaled)
        if probs.shape[1] == 1:
            preds = (probs > 0.5).astype(int).reshape(-1)
        else:
            preds = np.argmax(probs, axis=1)
        result_df = batch_df.copy()
        result_df['prediction'] = preds
        st.write("Predictions preview:")
        st.dataframe(result_df.head())
        csv_bytes = result_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download predictions CSV", data=csv_bytes, file_name="predictions.csv")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ — cute ANN Streamlit app. Customize colors, logos, or behaviors for your project presentation!")

