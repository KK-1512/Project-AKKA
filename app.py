import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ================================
# LOAD MODEL
# ================================
model_data = joblib.load("cvd_model.pkl")
model = model_data["model"]
FEATURES = model_data["features"]

st.set_page_config(page_title="CVD Hardness Predictor", layout="wide")

st.title("🔬 CVD TiN/TiON Hardness Prediction")
st.write("Predict hardness based on Temperature and Deposition Time")

# ================================
# INPUT SECTION
# ================================
col1, col2 = st.columns(2)

with col1:
    temp = st.number_input(
        "Temperature (°C)",
        min_value=400.0,
        max_value=800.0,
        value=600.0,
        step=10.0
    )

with col2:
    time = st.number_input(
        "Deposition Time (min)",
        min_value=30.0,
        max_value=180.0,
        value=90.0,
        step=5.0
    )

# ================================
# WARNING FOR OUT-OF-RANGE INPUT
# ================================
if temp < 500 or temp > 700 or time < 60 or time > 120:
    st.warning("⚠️ Input is outside the experimental training range (500–700°C, 60–120 min). Prediction may be unreliable.")
else:
    st.success("✅ Input is within the trained experimental range.")

# Feature engineering (same as training)
def create_features(temp, time):
    return pd.DataFrame({
        'Temperature_C': [temp],
        'Deposition_Time_min': [time],
        'Temp_x_Time': [temp * time],
        'Temp_squared': [temp**2],
        'Time_squared': [time**2]
    })

# ================================
# PREDICTION
# ================================
if st.button("Predict Hardness"):
    input_df = create_features(temp, time)
    prediction = model.predict(input_df)[0]

    st.success(f"Predicted Hardness: {prediction:.2f} HV")

# ================================
# OPTIONAL: LOAD DATA FOR GRAPHS
# ================================
try:
    df = pd.read_excel("Hardness.xlsx")

    st.subheader("📊 Data Visualization")

    col3, col4 = st.columns(2)

    with col3:
        fig, ax = plt.subplots()
        ax.scatter(df.iloc[:,0], df.iloc[:,2])
        ax.set_xlabel("Temperature")
        ax.set_ylabel("Hardness")
        st.pyplot(fig)

    with col4:
        fig, ax = plt.subplots()
        sns.boxplot(data=df, ax=ax)
        st.pyplot(fig)

    # Feature importance
    st.subheader("📌 Feature Importance")
    importance = model.feature_importances_
    fig, ax = plt.subplots()
    ax.bar(FEATURES, importance)
    plt.xticks(rotation=45)
    st.pyplot(fig)

except:
    st.warning("Dataset not found. Upload Hardness.xlsx for full visualization.")
