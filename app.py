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
    temp = st.slider("Temperature (°C)", 400, 800, 600)

with col2:
    time = st.slider("Deposition Time (min)", 30, 180, 90)

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
