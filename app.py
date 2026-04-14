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

st.title("🔬 CVD TiN/TiON Hardness Prediction System")

# ================================
# INPUT SECTION (FREE INPUT + RANGE)
# ================================
st.subheader("⚙️ Enter CVD Parameters")

col1, col2 = st.columns(2)

with col1:
    temp = st.number_input("Temperature (°C)", min_value=400.0, max_value=800.0, value=600.0)

with col2:
    time = st.number_input("Deposition Time (min)", min_value=30.0, max_value=180.0, value=90.0)

# ================================
# FEATURE ENGINEERING
# ================================
def create_features(temp, time):
    return pd.DataFrame({
        'Temperature_C': [temp],
        'Deposition_Time_min': [time],
        'Temp_x_Time': [temp * time],
        'Temp_squared': [temp**2],
        'Time_squared': [time**2]
    })

# ================================
# PREDICTION WITH RANGE
# ================================
if st.button("Predict Hardness"):

    input_df = create_features(temp, time)

    # Main prediction
    prediction = model.predict(input_df)[0]

    # Individual tree predictions (for range)
    tree_preds = np.array([tree.predict(input_df)[0] for tree in model.estimators_])

    min_val = tree_preds.min()
    max_val = tree_preds.max()
    std_dev = tree_preds.std()

    # ================================
    # OUTPUT
    # ================================
    st.success(f"🎯 Predicted Hardness: {prediction:.2f} HV")

    col3, col4, col5 = st.columns(3)

    col3.metric("Minimum Expected", f"{min_val:.2f} HV")
    col4.metric("Maximum Expected", f"{max_val:.2f} HV")
    col5.metric("Confidence Range (±)", f"{(1.96*std_dev):.2f} HV")

    st.info(f"📊 Estimated Range: {prediction - 1.96*std_dev:.2f} HV  to  {prediction + 1.96*std_dev:.2f} HV")

# ================================
# OPTIONAL VISUALIZATION
# ================================
st.subheader("📊 Data Visualization")

try:
    df = pd.read_excel("Hardness.xlsx")

    col6, col7 = st.columns(2)

    with col6:
        fig, ax = plt.subplots()
        ax.scatter(df.iloc[:,0], df.iloc[:,2])
        ax.set_xlabel("Temperature")
        ax.set_ylabel("Hardness")
        ax.set_title("Temperature vs Hardness")
        st.pyplot(fig)

    with col7:
        fig, ax = plt.subplots()
        sns.histplot(df.iloc[:,2], kde=True, ax=ax)
        ax.set_title("Hardness Distribution")
        st.pyplot(fig)

    # Feature importance
    st.subheader("📌 Feature Importance")

    importance = model.feature_importances_

    fig, ax = plt.subplots()
    ax.bar(FEATURES, importance)
    plt.xticks(rotation=45)
    ax.set_title("Feature Importance")
    st.pyplot(fig)

except:
    st.warning("Upload Hardness.xlsx to enable full visualization.")
