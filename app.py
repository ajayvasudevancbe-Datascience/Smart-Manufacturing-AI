
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import datetime

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Smart Manufacturing AI",
    page_icon="🏭",
    layout="wide"
)

st.title("🏭 Smart Manufacturing AI")
st.subheader("Machine Monitoring and Failure Prediction")

st.markdown("""
This application uses Machine Learning to predict possible
machine failures based on operating sensor values.
""")

st.divider()

# =====================================================
# 2. MODEL FILE PATHS
# =====================================================

MODEL_PATH = "smart_manufacturing_model/xgb_model.pkl"
PREPROCESSOR_PATH = "smart_manufacturing_model/preprocessor.pkl"

# =====================================================
# 3. LOAD MODEL AND PREPROCESSOR
# =====================================================

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor


if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

if not os.path.exists(PREPROCESSOR_PATH):
    st.error(f"Preprocessor file not found: {PREPROCESSOR_PATH}")
    st.stop()

try:
    model, preprocessor = load_model()
except Exception as error:
    st.error(f"Error loading model: {error}")
    st.stop()

# =====================================================
# 4. SESSION STATE – PREDICTION HISTORY
# =====================================================

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# =====================================================
# 5. SIDEBAR – MACHINE INPUTS
# =====================================================

st.sidebar.title("⚙️ Machine Configuration")

machine_type = st.sidebar.selectbox(
    "Machine Type",
    ["L", "M", "H"]
)

air_temp = st.sidebar.number_input(
    "Air Temperature (K)",
    min_value=290.0,
    max_value=310.0,
    value=300.0,
    step=0.1
)

process_temp = st.sidebar.number_input(
    "Process Temperature (K)",
    min_value=290.0,
    max_value=320.0,
    value=310.0,
    step=0.1
)

rotational_speed = st.sidebar.number_input(
    "Rotational Speed (rpm)",
    min_value=1000,
    max_value=3000,
    value=1500,
    step=10
)

torque = st.sidebar.number_input(
    "Torque (Nm)",
    min_value=0.0,
    max_value=100.0,
    value=40.0,
    step=0.5
)

tool_wear = st.sidebar.number_input(
    "Tool Wear (min)",
    min_value=0,
    max_value=300,
    value=100,
    step=1
)

# =====================================================
# 6. FEATURE ENGINEERING
# =====================================================

temperature_difference = (
    process_temp - air_temp
)

power = (
    2 * np.pi * rotational_speed * torque / 60
)

if tool_wear <= 50:
    tool_wear_category = "Low"
elif tool_wear <= 150:
    tool_wear_category = "Medium"
elif tool_wear <= 250:
    tool_wear_category = "High"
else:
    tool_wear_category = "Very High"

# =====================================================
# 7. CREATE INPUT DATAFRAME
# =====================================================

input_data = pd.DataFrame({
    "Type": [machine_type],
    "Air temperature [K]": [air_temp],
    "Process temperature [K]": [process_temp],
    "Rotational speed [rpm]": [rotational_speed],
    "Torque [Nm]": [torque],
    "Tool wear [min]": [tool_wear],
    "Temperature Difference [K]": [temperature_difference],
    "Power [W]": [power],
    "Tool Wear Category": [tool_wear_category]
})

# =====================================================
# 8. MACHINE MONITORING DASHBOARD
# =====================================================

st.header("📊 Machine Monitoring Dashboard")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Air Temperature",
    f"{air_temp:.2f} K"
)

col2.metric(
    "Process Temperature",
    f"{process_temp:.2f} K"
)

col3.metric(
    "Temperature Difference",
    f"{temperature_difference:.2f} K"
)

col4, col5, col6 = st.columns(3)

col4.metric(
    "Rotational Speed",
    f"{rotational_speed} rpm"
)

col5.metric(
    "Torque",
    f"{torque:.2f} Nm"
)

col6.metric(
    "Tool Wear",
    f"{tool_wear} min"
)

col7, col8 = st.columns(2)

col7.metric(
    "Calculated Power",
    f"{power:.2f} W"
)

col8.metric(
    "Tool Wear Category",
    tool_wear_category
)

with st.expander("View Machine Input Data"):
    st.dataframe(
        input_data,
        use_container_width=True
    )

st.divider()

# =====================================================
# 9. MACHINE FAILURE PREDICTION
# =====================================================

st.header("🤖 Machine Failure Prediction")

if st.button(
    "Predict Machine Failure",
    type="primary",
    use_container_width=True
):

    try:
        # Preprocess the input
        processed_data = preprocessor.transform(
            input_data
        )

        # Predict machine condition
        prediction = int(
            model.predict(processed_data)[0]
        )

        # Get prediction probabilities
        probabilities = model.predict_proba(
            processed_data
        )[0]

        normal_probability = float(
            probabilities[0] * 100
        )

        failure_probability = float(
            probabilities[1] * 100
        )

        # Determine machine status
        if prediction == 1:
            status = "Machine Failure Predicted"
        else:
            status = "Normal Machine Operation"

        # Save prediction
        result = {
            "Date and Time": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "Machine Type": machine_type,
            "Air Temperature": air_temp,
            "Process Temperature": process_temp,
            "Rotational Speed": rotational_speed,
            "Torque": torque,
            "Tool Wear": tool_wear,
            "Prediction": status,
            "Failure Probability (%)": round(
                failure_probability, 2
            )
        }

        st.session_state.prediction_history.append(
            result
        )

        # Display result
        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("⚠️ Machine Failure Predicted")
            st.warning(
                "The model predicts a possible machine failure. "
                "Inspect the machine and verify its operating condition."
            )
        else:
            st.success("✅ Normal Machine Operation")
            st.info(
                "The model predicts normal operation based on "
                "the supplied sensor values."
            )

        # Display probabilities
        result_col1, result_col2 = st.columns(2)

        result_col1.metric(
            "Failure Probability",
            f"{failure_probability:.2f}%"
        )

        result_col2.metric(
            "Normal Probability",
            f"{normal_probability:.2f}%"
        )

        # Probability chart
        probability_df = pd.DataFrame({
            "Machine Condition": [
                "Normal",
                "Failure"
            ],
            "Probability (%)": [
                normal_probability,
                failure_probability
            ]
        })

        fig = px.bar(
            probability_df,
            x="Machine Condition",
            y="Probability (%)",
            color="Machine Condition",
            text="Probability (%)",
            title="Machine Condition Prediction"
        )

        fig.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside"
        )

        fig.update_layout(
            yaxis_range=[0, 110]
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    except Exception as error:
        st.error(
            f"Prediction error: {error}"
        )

st.divider()

# =====================================================
# 10. PREDICTION HISTORY
# =====================================================

st.header("📋 Prediction History")

if st.session_state.prediction_history:

    history_df = pd.DataFrame(
        st.session_state.prediction_history
    )

    st.dataframe(
        history_df,
        use_container_width=True
    )

    # History chart
    history_chart = px.bar(
        history_df,
        x="Date and Time",
        y="Failure Probability (%)",
        color="Prediction",
        title="Failure Probability History"
    )

    st.plotly_chart(
        history_chart,
        use_container_width=True
    )

    # Download history
    csv_data = history_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="📥 Download Prediction History",
        data=csv_data,
        file_name="machine_prediction_history.csv",
        mime="text/csv"
    )

    if st.button("Clear Prediction History"):
        st.session_state.prediction_history = []
        st.rerun()

else:
    st.info(
        "No prediction history available. "
        "Enter machine values and run a prediction."
    )

st.divider()

# =====================================================
# 11. PROJECT INFORMATION
# =====================================================

st.header("📌 Project Information")

project_col1, project_col2 = st.columns(2)

with project_col1:
    st.markdown("""
    **Project:** Smart Manufacturing AI

    **Application:** Machine Monitoring and Failure Prediction

    **Dataset:** AI4I 2020 Predictive Maintenance Dataset

    **Target Variable:** Machine Failure
    """)

with project_col2:
    st.markdown("""
    **Machine Learning Model:** XGBoost

    **Deep Learning Model:** Artificial Neural Network

    **Programming Language:** Python

    **Dashboard:** Streamlit

    **Visualization:** Plotly
    """)

st.divider()

st.caption(
    "Educational demonstration using the AI4I 2020 synthetic dataset. "
    "This application is not a certified industrial safety system."
)
