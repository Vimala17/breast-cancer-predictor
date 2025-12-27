import streamlit as st
import tensorflow as tf
import pandas as pd
import pickle
import warnings

warnings.filterwarnings("ignore")

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="Breast Cancer Prediction",
    layout="wide",
)

# -------------------------------
# CLEAN DARK MEDICAL UI CSS
# -------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #020617, #0f172a);
    color: #e5e7eb;
}

h1 {
    text-align: center;
    color: #e5e7eb;
    font-weight: 800;
}

.subtitle {
    text-align: center;
    font-size: 17px;
    color: #cbd5f5;
    margin-bottom: 30px;
}

/* INPUT BOX */
[data-baseweb="input"] input {
    background-color: #020617 !important;
    color: #e5e7eb !important;
    border-radius: 8px;
}

/* 🔥 STREAMLIT BUTTON */
div.stButton > button {
    background: linear-gradient(90deg, #020617, #e5e7eb) !important;
    color: #ffffff !important;
    font-size: 20px !important;
    border-radius: 14px !important;
    padding: 14px !important;
    width: 100% !important;
    border: none !important;
}



/* RESULT BOX */
.result-box {
    margin-top: 25px;
    padding: 26px;
    border-radius: 16px;
    text-align: center;
    font-size: 26px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)


# -------------------------------
# Title
# -------------------------------
st.markdown("<h1>🩺 Breast Cancer Prediction System</h1>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>AI-based medical prediction system</div>",
    unsafe_allow_html=True
)

# -------------------------------
# Load model & scaler
# -------------------------------
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model("breast_cancer_model.h5")
    with open("BC_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_artifacts()

# -------------------------------
# Feature groups
# -------------------------------
feature_groups = {
    "🔬 Mean Features": [
        'radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean',
        'compactness_mean','concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean'
    ],
    "📐 SE Features": [
        'radius_se','texture_se','perimeter_se','area_se','smoothness_se',
        'compactness_se','concavity_se','concave points_se','symmetry_se','fractal_dimension_se'
    ],
    "⚠️ Worst Features": [
        'radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst',
        'compactness_worst','concavity_worst','concave points_worst','symmetry_worst','fractal_dimension_worst'
    ]
}

default_values = {
    'radius_mean':14.0,'texture_mean':20.0,'perimeter_mean':90.0,'area_mean':600.0,'smoothness_mean':0.1,
    'compactness_mean':0.15,'concavity_mean':0.2,'concave points_mean':0.1,'symmetry_mean':0.2,'fractal_dimension_mean':0.06,
    'radius_se':0.2,'texture_se':1.0,'perimeter_se':1.5,'area_se':20.0,'smoothness_se':0.005,
    'compactness_se':0.02,'concavity_se':0.03,'concave points_se':0.01,'symmetry_se':0.03,'fractal_dimension_se':0.004,
    'radius_worst':16.0,'texture_worst':25.0,'perimeter_worst':105.0,'area_worst':800.0,'smoothness_worst':0.12,
    'compactness_worst':0.2,'concavity_worst':0.3,'concave points_worst':0.15,'symmetry_worst':0.25,'fractal_dimension_worst':0.08
}

input_data = {}

# -------------------------------
# CLEAN FEATURES UI (TABS)
# -------------------------------
st.markdown("")

tabs = st.tabs(["🔬 Mean", "📐 Standard Error", "⚠️ Worst"])

with tabs[0]:
    cols = st.columns(2)
    for i, feature in enumerate(feature_groups["🔬 Mean Features"]):
        with cols[i % 2]:
            input_data[feature] = st.number_input(
                feature.replace("_", " ").title(),
                value=float(default_values[feature]),
                format="%.4f"
            )

with tabs[1]:
    cols = st.columns(2)
    for i, feature in enumerate(feature_groups["📐 SE Features"]):
        with cols[i % 2]:
            input_data[feature] = st.number_input(
                feature.replace("_", " ").title(),
                value=float(default_values[feature]),
                format="%.4f"
            )

with tabs[2]:
    cols = st.columns(2)
    for i, feature in enumerate(feature_groups["⚠️ Worst Features"]):
        with cols[i % 2]:
            input_data[feature] = st.number_input(
                feature.replace("_", " ").title(),
                value=float(default_values[feature]),
                format="%.4f"
            )

# -------------------------------
# Prediction
# -------------------------------
st.markdown("<div class='predict-btn'>", unsafe_allow_html=True)
predict = st.button(" Predict Cancer Type")
st.markdown("</div>", unsafe_allow_html=True)

if predict:
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0][0]

    if prediction > 0.5:
        st.markdown(
            "<div class='result-box' style='background:#7f1d1d;color:#fee2e2;'>🚨 Malignant Tumor</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div class='result-box' style='background:#14532d;color:#dcfce7;'>✅ Benign Tumor</div>",
            unsafe_allow_html=True
        )
