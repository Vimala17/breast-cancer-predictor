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
    layout="centered"
)

st.title("Breast Cancer Prediction")
st.write("Enter the tumor feature values and redict the class.")

# -------------------------------
# Load model & scaler (once)
# -------------------------------
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model("breast_cancer_model.h5")
    with open("BC_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_artifacts()

# -------------------------------
# Input fields (EXACT feature names)
# -------------------------------
st.subheader("Input Features")

input_data = {
    'radius_mean': st.number_input('radius_mean', value=14.0),
    'texture_mean': st.number_input('texture_mean', value=20.0),
    'perimeter_mean': st.number_input('perimeter_mean', value=90.0),
    'area_mean': st.number_input('area_mean', value=600.0),
    'smoothness_mean': st.number_input('smoothness_mean', value=0.1),
    'compactness_mean': st.number_input('compactness_mean', value=0.15),
    'concavity_mean': st.number_input('concavity_mean', value=0.2),
    'concave points_mean': st.number_input('concave points_mean', value=0.1),
    'symmetry_mean': st.number_input('symmetry_mean', value=0.2),
    'fractal_dimension_mean': st.number_input('fractal_dimension_mean', value=0.06),

    'radius_se': st.number_input('radius_se', value=0.2),
    'texture_se': st.number_input('texture_se', value=1.0),
    'perimeter_se': st.number_input('perimeter_se', value=1.5),
    'area_se': st.number_input('area_se', value=20.0),
    'smoothness_se': st.number_input('smoothness_se', value=0.005),
    'compactness_se': st.number_input('compactness_se', value=0.02),
    'concavity_se': st.number_input('concavity_se', value=0.03),
    'concave points_se': st.number_input('concave points_se', value=0.01),
    'symmetry_se': st.number_input('symmetry_se', value=0.03),
    'fractal_dimension_se': st.number_input('fractal_dimension_se', value=0.004),

    'radius_worst': st.number_input('radius_worst', value=16.0),
    'texture_worst': st.number_input('texture_worst', value=25.0),
    'perimeter_worst': st.number_input('perimeter_worst', value=105.0),
    'area_worst': st.number_input('area_worst', value=800.0),
    'smoothness_worst': st.number_input('smoothness_worst', value=0.12),
    'compactness_worst': st.number_input('compactness_worst', value=0.2),
    'concavity_worst': st.number_input('concavity_worst', value=0.3),
    'concave points_worst': st.number_input('concave points_worst', value=0.15),
    'symmetry_worst': st.number_input('symmetry_worst', value=0.25),
    'fractal_dimension_worst': st.number_input('fractal_dimension_worst', value=0.08),
}

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict"):
    input_df = pd.DataFrame([input_data])

    # Safety check — do NOT remove
    if list(input_df.columns) != list(scaler.feature_names_in_):
        st.error("Feature mismatch between input and trained scaler.")
        st.stop()

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0][0]

    predicted_class = "Malignant" if prediction > 0.5 else "Benign"

    st.subheader("Result")
    st.write(f"**Prediction:** {predicted_class}")
    st.write(f"**Probability:** {prediction:.4f}")