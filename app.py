
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Smart Manufacturing AI",
    page_icon="🏭",
    layout="wide"
)

# --------------------------------------------------
# LOAD MODEL AND PREPROCESSOR
# --------------------------------------------------

MODEL_PATH = "smart_manufacturing_model/xgb_model.pkl"
PREPROCESSOR_PATH = "smart_manufacturing_model/preprocessor.pkl"

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor

if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
    st.error(
        "Model files not found. Place xgb_model.pkl and "
        "preprocessor.pkl inside smart_manufacturing_model."
    )
    st.stop()

model, preprocessor = load_artifacts()

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🏭 Smart Manufacturing AI")
st.subheader("Machine Monitoring and Failure Prediction")

st.markdown(
    "An AI-powered predictive maintenance application "
    "using Machine Learning and XGBoost."
)

st.divider()

# --------------------------------------------------
# SIDEBAR INPUTS
# --------------------------------------------------

st.sidebar.header("⚙️ Machine Sensor Inputs")

machine_type = st.sidebar.selectbox(
    "Machine Type",
    ["L", "M", "H"]
)

air_temp = st.sidebar.number_input(
    "Air Temperature (K)",
    min_value=250.0,
    max_value=350.0,
    value=298.0
)

process_temp = st.sidebar.number_input(
    "Process Temperature (K)",
    min_value=250.0,
    max_value=400.0,
    value=308.0
)

rotational_speed = st.sidebar.number_input(
    "Rotational Speed (rpm)",
    min_value=500,
    max_value=4000,
    value=1500
)

torque = st.sidebar.number_input(
    "Torque (Nm)",
    min_value=1.0,
    max_value=100.0,
    value=40.0
)

tool_wear = st.sidebar.number_input(
    "Tool Wear (min)",
    min_value=0,
    max_value=300,
    value=100
)

# --------------------------------------------------
# FEATURE ENGINEERING
# --------------------------------------------------

temperature_difference = process_temp - air_temp

power = (
    2 * np.pi * rotational_speed * torque / 60
)

if tool_wear <= 50:
    wear_category = "Low"
elif tool_wear <= 150:
    wear_category = "Medium"
elif tool_wear <= 250:
    wear_category = "High"
else:
    wear_category = "Very High"

# --------------------------------------------------
# MACHINE INPUT DATA
# --------------------------------------------------

input_data = pd.DataFrame([{
    "Type": machine_type,
    "Air temperature [K]": air_temp,
    "Process temperature [K]": process_temp,
    "Rotational speed [rpm]": rotational_speed,
    "Torque [Nm]": torque,
    "Tool wear [min]": tool_wear,
    "Temperature Difference [K]": temperature_difference,
    "Power [W]": power,
    "Tool Wear Category": wear_category
}])

# --------------------------------------------------
# MACHINE CONDITION
# --------------------------------------------------

st.header("📊 Machine Condition Monitoring")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Air Temperature", f"{air_temp:.2f} K")
col2.metric("Process Temperature", f"{process_temp:.2f} K")
col3.metric("Rotational Speed", f"{rotational_speed} rpm")
col4.metric("Tool Wear", f"{tool_wear} min")

# --------------------------------------------------
# FEATURE VALUES
# --------------------------------------------------

with st.expander("View Calculated Machine Features"):
    st.dataframe(input_data, use_container_width=True)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

st.divider()
st.header("🤖 Machine Failure Prediction")

if st.button("Predict Machine Failure", type="primary"):

    try:
        # Apply the same fitted preprocessing used in training
        processed_input = preprocessor.transform(input_data)

        prediction = int(model.predict(processed_input)[0])

        probability = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(processed_input)
            probability = float(probabilities[0][1])

        if prediction == 1:
            st.error("⚠️ Machine Failure Predicted")
            st.warning(
                "The model predicts a possible machine failure. "
                "Inspect the machine and schedule maintenance."
            )
        else:
            st.success("✅ Machine Failure Not Predicted")
            st.info(
                "The model predicts normal operation. "
                "Continue routine machine monitoring."
            )

        if probability is not None:
            st.metric(
                "Predicted Failure Probability",
                f"{probability * 100:.2f}%"
            )

            probability_chart = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=probability * 100,
                    title={"text": "Failure Probability (%)"},
                    gauge={"axis": {"range": [0, 100]}}
                )
            )

            probability_chart.update_layout(height=300)
            st.plotly_chart(
                probability_chart,
                use_container_width=True
            )

        # Save prediction history
        st.session_state.history.append({
            "Machine Type": machine_type,
            "Air Temperature": air_temp,
            "Process Temperature": process_temp,
            "Rotational Speed": rotational_speed,
            "Torque": torque,
            "Tool Wear": tool_wear,
            "Prediction": "Failure" if prediction == 1 else "Normal",
            "Failure Probability": (
                probability * 100 if probability is not None else None
            )
        })

    except Exception as error:
        st.error(f"Prediction error: {error}")
        st.info(
            "Check that the input feature names, order, "
            "and preprocessing object match your training workflow."
        )

# --------------------------------------------------
# PREDICTION HISTORY
# --------------------------------------------------

st.divider()
st.header("📋 Prediction History")

if st.session_state.history:

    history_df = pd.DataFrame(st.session_state.history)

    st.dataframe(
        history_df,
        use_container_width=True
    )

    csv = history_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Prediction History",
        data=csv,
        file_name="machine_prediction_history.csv",
        mime="text/csv"
    )

    history_counts = history_df["Prediction"].value_counts()

    history_chart_df = history_counts.rename_axis(
        "Machine Status"
    ).reset_index(name="Count")

    fig = px.pie(
        history_chart_df,
        names="Machine Status",
        values="Count",
        hole=0.4,
        title="Prediction History Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

    if st.button("Clear Prediction History"):
        st.session_state.history = []
        st.rerun()

else:
    st.info("No predictions have been recorded yet.")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Smart Manufacturing AI | Predictive Maintenance "
    "| XGBoost Machine Learning"
)

st.caption(
    "Educational project using the AI4I 2020 synthetic dataset. "
    "Predictions should not replace professional machine inspection."
)
