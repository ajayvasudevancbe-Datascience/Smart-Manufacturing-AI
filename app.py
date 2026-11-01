
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

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

st.write(
    "An AI-powered predictive maintenance application "
    "that predicts possible machine failures using sensor data."
)

# =====================================================
# 2. PATHS AND DATASET
# =====================================================

MODEL_DIR = "smart_manufacturing_model"
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_model.pkl")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.pkl")

DATA_URL = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/00601/ai4i2020.csv"
)

os.makedirs(MODEL_DIR, exist_ok=True)

# =====================================================
# 3. FEATURE ENGINEERING
# =====================================================

def create_features(data):
    data = data.copy()

    data["Temperature Difference [K]"] = (
        data["Process temperature [K]"]
        - data["Air temperature [K]"]
    )

    data["Power [W]"] = (
        2 * np.pi
        * data["Rotational speed [rpm]"]
        * data["Torque [Nm]"] / 60
    )

    data["Tool Wear Category"] = pd.cut(
        data["Tool wear [min]"],
        bins=[-1, 50, 150, 250, np.inf],
        labels=["Low", "Medium", "High", "Very High"]
    ).astype(str)

    return data


FEATURE_COLUMNS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Temperature Difference [K]",
    "Power [W]",
    "Tool Wear Category"
]

NUMERIC_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Temperature Difference [K]",
    "Power [W]"
]

CATEGORICAL_COLUMNS = [
    "Type",
    "Tool Wear Category"
]

# =====================================================
# 4. TRAIN MODEL IF FILES ARE MISSING
# =====================================================

@st.cache_resource
def load_or_train_model():

    if (
        os.path.exists(MODEL_PATH)
        and os.path.exists(PREPROCESSOR_PATH)
    ):
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)

        return model, preprocessor, "Loaded saved model"

    # Download dataset
    df = pd.read_csv(DATA_URL)

    # Feature engineering
    df = create_features(df)

    # Select features and target
    X = df[FEATURE_COLUMNS]
    y = df["Machine failure"]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Preprocessing
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_COLUMNS),
            ("categorical", categorical_transformer, CATEGORICAL_COLUMNS)
        ]
    )

    # Fit preprocessing on training data only
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Handle class imbalance through class weighting
    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=negative_count / max(positive_count, 1),
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    # Train model
    model.fit(X_train_processed, y_train)

    # Evaluate on original test set
    predictions = model.predict(X_test_processed)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    # Save both files
    joblib.dump(model, MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }

    return model, preprocessor, metrics


# =====================================================
# 5. LOAD OR TRAIN
# =====================================================

with st.spinner("Loading model or training XGBoost model..."):

    try:
        model, preprocessor, model_status = load_or_train_model()

        st.success("Model is ready!")

        if isinstance(model_status, dict):
            st.subheader("Model Evaluation")

            metric_cols = st.columns(4)

            for col, (name, value) in zip(
                metric_cols, model_status.items()
            ):
                col.metric(name, f"{value:.2%}")

        else:
            st.info(model_status)

    except Exception as error:
        st.error(f"Model setup failed: {error}")
        st.stop()


# =====================================================
# 6. SESSION STATE
# =====================================================

if "history" not in st.session_state:
    st.session_state.history = []


# =====================================================
# 7. SIDEBAR INPUTS
# =====================================================

st.sidebar.header("⚙️ Machine Sensor Inputs")

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
# 8. CREATE INPUT DATA
# =====================================================

input_data = pd.DataFrame({
    "Type": [machine_type],
    "Air temperature [K]": [air_temp],
    "Process temperature [K]": [process_temp],
    "Rotational speed [rpm]": [rotational_speed],
    "Torque [Nm]": [torque],
    "Tool wear [min]": [tool_wear]
})

input_data = create_features(input_data)
input_data = input_data[FEATURE_COLUMNS]


# =====================================================
# 9. MACHINE MONITORING
# =====================================================

st.header("📊 Machine Monitoring Dashboard")

col1, col2, col3 = st.columns(3)

col1.metric("Air Temperature", f"{air_temp:.2f} K")
col2.metric("Process Temperature", f"{process_temp:.2f} K")
col3.metric(
    "Temperature Difference",
    f"{process_temp - air_temp:.2f} K"
)

col4, col5, col6 = st.columns(3)

col4.metric("Rotational Speed", f"{rotational_speed} rpm")
col5.metric("Torque", f"{torque:.2f} Nm")
col6.metric("Tool Wear", f"{tool_wear} min")

col7, col8 = st.columns(2)

power_value = (
    2 * np.pi * rotational_speed * torque / 60
)

col7.metric("Calculated Power", f"{power_value:.2f} W")
col8.metric("Tool Wear Category", input_data.iloc[0]["Tool Wear Category"])

with st.expander("View Machine Input Data"):
    st.dataframe(input_data, use_container_width=True)


# =====================================================
# 10. MACHINE FAILURE PREDICTION
# =====================================================

st.divider()
st.header("🤖 Machine Failure Prediction")

if st.button(
    "Predict Machine Failure",
    type="primary",
    use_container_width=True
):

    try:
        processed_input = preprocessor.transform(input_data)

        prediction = int(model.predict(processed_input)[0])

        probabilities = model.predict_proba(processed_input)[0]

        normal_probability = float(probabilities[0] * 100)
        failure_probability = float(probabilities[1] * 100)

        status = (
            "Machine Failure Predicted"
            if prediction == 1
            else "Normal Machine Operation"
        )

        # Save history
        result = {
            "Time": pd.Timestamp.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "Machine Type": machine_type,
            "Air Temperature (K)": air_temp,
            "Process Temperature (K)": process_temp,
            "Rotational Speed (rpm)": rotational_speed,
            "Torque (Nm)": torque,
            "Tool Wear (min)": tool_wear,
            "Prediction": status,
            "Failure Probability (%)": round(
                failure_probability, 2
            )
        }

        st.session_state.history.append(result)

        # Show result
        if prediction == 1:
            st.error("⚠️ Machine Failure Predicted")
            st.warning(
                "The model predicts a possible machine failure. "
                "Inspect the machine and verify its condition."
            )
        else:
            st.success("✅ Normal Machine Operation")
            st.info(
                "The model predicts normal operation for the "
                "provided sensor values."
            )

        # Show probabilities
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
            "Condition": ["Normal", "Failure"],
            "Probability": [
                normal_probability,
                failure_probability
            ]
        })

        fig = px.bar(
            probability_df,
            x="Condition",
            y="Probability",
            color="Condition",
            text_auto=".2f",
            title="Machine Failure Probability"
        )

        fig.update_layout(yaxis_range=[0, 100])

        st.plotly_chart(fig, use_container_width=True)

    except Exception as error:
        st.error(f"Prediction failed: {error}")


# =====================================================
# 11. PREDICTION HISTORY
# =====================================================

st.divider()
st.header("📋 Prediction History")

if st.session_state.history:

    history_df = pd.DataFrame(st.session_state.history)

    st.dataframe(
        history_df,
        use_container_width=True
    )

    history_chart = px.bar(
        history_df,
        x="Time",
        y="Failure Probability (%)",
        color="Prediction",
        title="Prediction History"
    )

    st.plotly_chart(
        history_chart,
        use_container_width=True
    )

    csv_file = history_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "📥 Download Prediction History",
        data=csv_file,
        file_name="machine_prediction_history.csv",
        mime="text/csv"
    )

    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()

else:
    st.info("No predictions have been recorded yet.")


# =====================================================
# 12. PROJECT INFORMATION
# =====================================================

st.divider()
st.header("📌 Project Information")

info1, info2 = st.columns(2)

with info1:
    st.markdown("""
    **Project:** Smart Manufacturing AI

    **Application:** Machine Monitoring and Failure Prediction

    **Dataset:** AI4I 2020 Predictive Maintenance Dataset

    **Target:** Machine Failure
    """)

with info2:
    st.markdown("""
    **Machine Learning:** XGBoost

    **Deep Learning:** Artificial Neural Network (developed separately)

    **Language:** Python

    **Dashboard:** Streamlit

    **Visualization:** Plotly
    """)

st.caption(
    "Educational demonstration using a synthetic predictive "
    "maintenance dataset. This is not a certified industrial "
    "safety or machine-control system."
)
