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

    st.subheader("Dataset Analysis")

    try:
        df = pd.read_excel("Hardness.xlsx")
        df.columns = ['Temperature_C', 'Time', 'Hardness']

        # Scatter
        col3, col4 = st.columns(2)

        with col3:
            fig, ax = plt.subplots()
            ax.scatter(df['Temperature_C'], df['Hardness'])
            ax.set_title("Temperature vs Hardness")
            st.pyplot(fig)

        with col4:
            fig, ax = plt.subplots()
            ax.scatter(df['Time'], df['Hardness'])
            ax.set_title("Time vs Hardness")
            st.pyplot(fig)

        # Distribution
        st.subheader("Hardness Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df['Hardness'], kde=True, ax=ax)
        st.pyplot(fig)

        # Boxplot
        st.subheader("Boxplot")
        fig, ax = plt.subplots()
        sns.boxplot(data=df, ax=ax)
        st.pyplot(fig)

        # Heatmap
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(), annot=True, ax=ax)
        st.pyplot(fig)

        # Feature Importance
        st.subheader("Feature Importance")
        importance = model.feature_importances_

        fig, ax = plt.subplots()
        ax.bar(FEATURES, importance)
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Actual vs Predicted
        X = df[['Temperature_C', 'Time']]
        y = df['Hardness']

        # recreate features
        X_full = pd.DataFrame({
            'Temperature_C': df['Temperature_C'],
            'Deposition_Time_min': df['Time'],
            'Temp_x_Time': df['Temperature_C'] * df['Time'],
            'Temp_squared': df['Temperature_C']**2,
            'Time_squared': df['Time']**2
        })

        y_pred = model.predict(X_full)

        st.subheader("Actual vs Predicted")
        fig, ax = plt.subplots()
        ax.scatter(y, y_pred)
        st.pyplot(fig)

        # Residual
        st.subheader("Residual Plot")
        residuals = y - y_pred
        fig, ax = plt.subplots()
        ax.scatter(y_pred, residuals)
        ax.axhline(0, color='red')
        st.pyplot(fig)

    except:
        st.error("Upload Hardness.xlsx to view analysis")
