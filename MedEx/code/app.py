# app.py
# Simple Streamlit app for Insurance Charges prediction
# Assumes final_model.pkl and final_scaler.pkl are in the same folder.

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# ---------------------------
# 1) Load saved artifacts
# ---------------------------
@st.cache_resource  # cache for faster load in Streamlit
def load_artifacts():
    # Update filenames if different
    with open("final_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("final_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_artifacts()

# ---------------------------
# 2) App header
# ---------------------------
st.title("Medical Insurance Charges Predictor")
st.write("Predict expected medical insurance charges (based on your trained model).")
st.write("---")

# ---------------------------
# 3) User inputs (interactive)
# ---------------------------
st.header("Enter patient details")

# Use same feature order as training:
# features = ['age','sex','bmi','children','smoker','region_northwest','region_southeast','region_southwest']

age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
sex = st.radio("Sex", options=["Female", "Male"], index=0)
bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=28.0, step=0.1, format="%.1f")
children = st.selectbox("Number of children", options=[0,1,2,3,4,5], index=0)
smoker = st.radio("Smoker", options=["No", "Yes"], index=0)

st.write("Region (choose the region):")
region = st.selectbox("Region", options=["northeast", "northwest", "southeast", "southwest"], index=0)

# ---------------------------
# 4) Convert inputs to model features
# ---------------------------
# Binary encode sex and smoker exactly like training:
sex_encoded = 1 if sex == "Male" else 0
smoker_encoded = 1 if smoker == "Yes" else 0

# One-hot for regions (drop_first=True used in training -> region_northwest, region_southeast, region_southwest)
region_northwest = 1 if region == "northwest" else 0
region_southeast = 1 if region == "southeast" else 0
region_southwest = 1 if region == "southwest" else 0

# Build single-row DataFrame in same order as training features
features = ['age','sex','bmi','children','smoker','region_northwest','region_southeast','region_southwest']

row = pd.DataFrame([[
    age,
    sex_encoded,
    bmi,
    children,
    smoker_encoded,
    region_northwest,
    region_southeast,
    region_southwest
]], columns=features)

st.subheader("Feature vector")
st.dataframe(row.T.rename(columns={0:"value"}))

# ---------------------------
# 5) Scale and predict
# ---------------------------
if st.button("Predict charges"):
    # Scale features using saved scaler
    X_scaled = scaler.transform(row)
    # Predict log_charges (model trained on log target)
    pred_log = model.predict(X_scaled)[0]
    # Convert back to original charges
    pred_charges = np.exp(pred_log)
    
    st.success(f"Predicted annual charges (approx): ₹ {pred_charges:,.2f}")
    st.write(f"Predicted log_charges: {pred_log:.4f}")
    
    # Extra: show simple bar of contributions (optional)
    st.subheader("Quick explanation (top features influence)")
    # We can't easily show SHAP here without extra deps; show a simple table of values instead
    st.write("Input values used for prediction:")
    st.table(row)

    # Plot: show where this prediction lies in distribution (if you upload CSV)
    try:
        df = pd.read_csv("cleaned_medex_data.csv")
        fig, ax = plt.subplots(figsize=(6,3))
        sns.histplot(df['charges'], bins=40, kde=True, ax=ax)
        ax.axvline(pred_charges, color='red', linestyle='--', label='Predicted charges')
        ax.legend()
        st.pyplot(fig)
    except Exception:
        # file not present — skip silently
        pass

st.write("---")
st.caption("Model trained on log(charges) — prediction converted back with exp(). For production, include validation and error handling.")
