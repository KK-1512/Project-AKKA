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

st.title("🔬 CVD TiN/TiON Hardness Prediction Dashboard")

# ================================
# TABS
# ================================
tab1, tab2 = st.tabs(["🔮 Prediction", "📊 Analysis"])

# ================================
# TAB 1: PREDICTION
# ================================
with tab1:

    st.subheader("Enter CVD Parameters")

    col1, col2 = st.columns(2)

    with col1:
        temp = st.number_input("Temperature (°C)", 400.0, 800.0, 600.0)

    with col2:
        time = st.number_input("Deposition Time (min)", 30.0, 180.0, 90.0)

    # Warning
    if temp < 500 or temp > 700 or time < 60 or time > 120:
        st.warning("⚠️ Outside training range (500–700°C, 60–120 min)")
    else:
        st.success("✅ Within training range")

    def create_features(temp, time):
        return pd.DataFrame({
            'Temperature_C': [temp],
            'Deposition_Time_min': [time],
            'Temp_x_Time': [temp * time],
            'Temp_squared': [temp**2],
            'Time_squared': [time**2]
        })

    if st.button("Predict Hardness"):
        input_df = create_features(temp, time)
        pred = model.predict(input_df)[0]

        # Range using trees
        tree_preds = np.array([t.predict(input_df)[0] for t in model.estimators_])

        st.success(f"Predicted Hardness: {pred:.2f} HV")
        st.info(f"Range: {tree_preds.min():.2f} HV to {tree_preds.max():.2f} HV")

# ================================
# TAB 2: ANALYSIS
# ================================
with tab2:

    st.subheader("📊 Dataset Analysis")

    try:
        df = pd.read_excel("Hardness.xlsx")
        df.columns = ['Temperature_C', 'Time', 'Hardness']

        # ================================
        # ROW 1 (2 GRAPHS SIDE BY SIDE)
        # ================================
        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(5,4))
            ax.scatter(df['Temperature_C'], df['Hardness'])
            ax.set_title("Temp vs Hardness")
            ax.set_xlabel("Temp")
            ax.set_ylabel("Hardness")
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(5,4))
            ax.scatter(df['Time'], df['Hardness'])
            ax.set_title("Time vs Hardness")
            st.pyplot(fig)

        # ================================
        # ROW 2
        # ================================
        col3, col4 = st.columns(2)

        with col3:
            fig, ax = plt.subplots(figsize=(5,4))
            sns.histplot(df['Hardness'], kde=True, ax=ax)
            ax.set_title("Hardness Distribution")
            st.pyplot(fig)

        with col4:
            fig, ax = plt.subplots(figsize=(5,4))
            sns.boxplot(data=df, ax=ax)
            ax.set_title("Boxplot")
            st.pyplot(fig)

        # ================================
        # ROW 3
        # ================================
        col5, col6 = st.columns(2)

        with col5:
            fig, ax = plt.subplots(figsize=(5,4))
            sns.heatmap(df.corr(), annot=True, ax=ax)
            ax.set_title("Correlation")
            st.pyplot(fig)

        with col6:
            importance = model.feature_importances_
            fig, ax = plt.subplots(figsize=(5,4))
            ax.bar(FEATURES, importance)
            ax.set_title("Feature Importance")
            plt.xticks(rotation=30)
            st.pyplot(fig)

        # ================================
        # ROW 4
        # ================================
        col7, col8 = st.columns(2)

        # Create features again
        X_full = pd.DataFrame({
            'Temperature_C': df['Temperature_C'],
            'Deposition_Time_min': df['Time'],
            'Temp_x_Time': df['Temperature_C'] * df['Time'],
            'Temp_squared': df['Temperature_C']**2,
            'Time_squared': df['Time']**2
        })

        y = df['Hardness']
        y_pred = model.predict(X_full)

        with col7:
            fig, ax = plt.subplots(figsize=(5,4))
            ax.scatter(y, y_pred)
            ax.set_title("Actual vs Predicted")
            st.pyplot(fig)

        with col8:
            residuals = y - y_pred
            fig, ax = plt.subplots(figsize=(5,4))
            ax.scatter(y_pred, residuals)
            ax.axhline(0, color='red')
            ax.set_title("Residual Plot")
            st.pyplot(fig)

    except:
        st.error("Upload Hardness.xlsx to view analysis")
