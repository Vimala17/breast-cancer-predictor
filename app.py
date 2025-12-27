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
    page_icon="🎗️"
)

# -------------------------------
# Custom CSS
# -------------------------------
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right, #ffe6f0, #e6f7ff);
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        font-size: 18px;
        border-radius: 10px;
        padding: 10px 20px;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #ff0000;
        color: #fff;
    }
    .input-card {
        padding: 15px;
        border-radius: 15px;
        margin-bottom: 15px;
        box-shadow: 3px 3px 12px #aaaaaa;
    }
    .mean-card {background: #ffd6e0;}
    .se-card {background: #d6f0ff;}
    .worst-card {background: #d6ffd6;}
    .prediction-box {
        border-radius: 15px;
        padding: 20px;
        text-align:center;
        font-size:24px;
        font-weight:bold;
        margin-top:10px;
        box-shadow: 3px 3px 12px #aaaaaa;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# Title
# -------------------------------
st.markdown("<h1 style='text-align:center; color:#ff4b4b;'>🎗️ Breast Cancer Prediction 🎗️</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:18px;'>Use the sliders below to enter tumor features.</p>", unsafe_allow_html=True)

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
# Feature Groups
# -------------------------------
feature_groups = {
    "Mean Features": [
        'radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean'
    ],
    "Standard Error (SE) Features": [
        'radius_se','texture_se','perimeter_se','area_se','smoothness_se','compactness_se','concavity_se','concave points_se','symmetry_se','fractal_dimension_se'
    ],
    "Worst Features": [
        'radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst','compactness_worst','concavity_worst','concave points_worst','symmetry_worst','fractal_dimension_worst'
    ]
}

default_values = {
    'radius_mean':14.0,'texture_mean':20.0,'perimeter_mean':90.0,'area_mean':600.0,'smoothness_mean':0.1,'compactness_mean':0.15,'concavity_mean':0.2,'concave points_mean':0.1,'symmetry_mean':0.2,'fractal_dimension_mean':0.06,
    'radius_se':0.2,'texture_se':1.0,'perimeter_se':1.5,'area_se':20.0,'smoothness_se':0.005,'compactness_se':0.02,'concavity_se':0.03,'concave points_se':0.01,'symmetry_se':0.03,'fractal_dimension_se':0.004,
    'radius_worst':16.0,'texture_worst':25.0,'perimeter_worst':105.0,'area_worst':800.0,'smoothness_worst':0.12,'compactness_worst':0.2,'concavity_worst':0.3,'concave points_worst':0.15,'symmetry_worst':0.25,'fractal_dimension_worst':0.08
}

input_data = {}

# Display sliders in colored cards per group
for group_name, group_features in feature_groups.items():
    card_class = "mean-card" if "Mean" in group_name else ("se-card" if "SE" in group_name else "worst-card")
    st.markdown(f"<div class='input-card {card_class}'>", unsafe_allow_html=True)
    st.markdown(f"### {group_name}")
    cols = st.columns(2)
    for i, feature in enumerate(group_features):
        col = cols[i % 2]
        input_data[feature] = col.slider(
            feature,
            min_value=float(default_values[feature]*0.5),
            max_value=float(default_values[feature]*1.5),
            value=float(default_values[feature]),
            step=0.01
        )
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict"):
    input_df = pd.DataFrame([input_data])
    if list(input_df.columns) != list(scaler.feature_names_in_):
        st.error("Feature mismatch with trained scaler!")
        st.stop()
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0][0]
    predicted_class = "Malignant" if prediction>0.5 else "Benign"
    
    color = "#ff4b4b" if predicted_class=="Malignant" else "#28a745"
    st.markdown(f"<div class='prediction-box' style='background-color:{color}; color:white;'>{predicted_class} ({prediction:.2f})</div>", unsafe_allow_html=True)
