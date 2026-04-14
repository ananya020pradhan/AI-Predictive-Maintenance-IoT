# ==============================
# AI Predictive Maintenance - Streamlit App
# ==============================

import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ------------------------------
# Load trained model & scaler & features
# ------------------------------
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")
features = joblib.load("models/features.pkl")

# ------------------------------
# UI CONFIG
# ------------------------------
st.set_page_config(page_title="Predictive Maintenance AI", layout="centered")

st.title("🔧 AI-Powered Predictive Maintenance System")
st.markdown("Enter machine sensor values to predict failure risk")

# ------------------------------
# INPUTS (only main real sensors)
# ------------------------------
temperature = st.slider("Temperature", 0, 150, 50)
vibration = st.slider("Vibration", 0.0, 5.0, 1.0)
pressure = st.slider("Pressure", 0, 300, 100)

# ------------------------------
# PREDICTION BUTTON
# ------------------------------
if st.button("Predict Machine Status"):

    # Create empty dataframe with SAME features used in training
    input_data = pd.DataFrame(columns=features)
    input_data.loc[0] = 0

    # Fill known sensor values
    if "temperature" in input_data.columns:
        input_data["temperature"] = temperature

    if "vibration" in input_data.columns:
        input_data["vibration"] = vibration

    if "pressure" in input_data.columns:
        input_data["pressure"] = pressure

    # ------------------------------
    # Scale input
    # ------------------------------
    data_scaled = scaler.transform(input_data)

    # ------------------------------
    # Prediction
    # ------------------------------
    prediction = model.predict(data_scaled)[0]
    probability = model.predict_proba(data_scaled)[0][1]

    # ------------------------------
    # OUTPUT
    # ------------------------------
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ HIGH RISK: Machine Failure Likely ({probability*100:.2f}%)")
    else:
        st.success(f"✅ MACHINE SAFE ({(1-probability)*100:.2f}%)")

    # ------------------------------
    # PROBABILITY BAR
    # ------------------------------
    st.subheader("Failure Probability")

    st.progress(float(probability))

    # ------------------------------
    # SIMPLE VISUALIZATION
    # ------------------------------
    st.subheader("Risk Breakdown")

    chart_data = pd.DataFrame({
        "Status": ["Safe", "Failure"],
        "Probability": [1 - probability, probability]
    })

    st.bar_chart(chart_data.set_index("Status"))

# ------------------------------
# FOOTER
# ------------------------------
st.markdown("---")
st.caption("Built with Machine Learning + IoT Simulation | Predictive Maintenance System")