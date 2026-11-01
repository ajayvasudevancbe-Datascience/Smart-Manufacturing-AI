
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from xgboost import XGBClassifier

# ==========================================================
# 1. PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Smart Manufacturing AI",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# 2. CUSTOM CSS
# ==========================================================

st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

.dashboard-title {
    font-size: 32px;
    font-weight: 800;
    color: #17365d;
}

.dashboard-subtitle {
    color: #64748b;
    font-size: 15px;
}

div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e2e8f0;
    padding: 18px;
    border-radius: 12px;
}

div.stButton > button {
    border-radius: 9px;
    font-weight: 600;
    min-height: 42px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# 3. PROJECT CONFIGURATION
# ==========================================================

MODEL_DIR = "smart_manufacturing_model"

MODEL_PATH = os.path.join(
    MODEL_DIR, "xgb_model.pkl"
)

PREPROCESSOR_PATH = os.path.join(
    MODEL_DIR, "preprocessor.pkl"
)

DATA_URL = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/00601/ai4i2020.csv"
)

os.makedirs(MODEL_DIR, exist_ok=True)

# ==========================================================
# 4. FEATURE ENGINEERING
# ==========================================================

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


# ==========================================================
# 5. MODEL TRAINING AND LOADING
# ==========================================================

@st.cache_resource
def load_or_train_model():

    # Load existing model
    if (
        os.path.exists(MODEL_PATH)
        and os.path.exists(PREPROCESSOR_PATH)
    ):

        model = joblib.load(MODEL_PATH)

        preprocessor = joblib.load(
            PREPROCESSOR_PATH
        )

        return model, preprocessor, None

    # Download dataset if model files are missing
    df = pd.read_csv(DATA_URL)

    df = create_features(df)

    X = df[FEATURE_COLUMNS]

    y = df["Machine failure"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_transformer, NUMERIC_COLUMNS),
        ("categorical", categorical_transformer, CATEGORICAL_COLUMNS)
    ])

    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    X_test_processed = preprocessor.transform(
        X_test
    )

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

    model.fit(
        X_train_processed,
        y_train
    )

    predictions = model.predict(
        X_test_processed
    )

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, zero_division=0
        )
    }

    joblib.dump(model, MODEL_PATH)

    joblib.dump(
        preprocessor,
        PREPROCESSOR_PATH
    )

    return model, preprocessor, metrics


# ==========================================================
# 6. LOAD MODEL
# ==========================================================

st.title("🏭 Smart Manufacturing AI")

st.caption(
    "Machine Monitoring | Predictive Maintenance | AI Analytics"
)

with st.spinner("Loading or training the machine learning model..."):

    try:

        model, preprocessor, model_metrics = (
            load_or_train_model()
        )

    except Exception as error:

        st.error(
            f"Unable to load or train model: {error}"
        )

        st.info(
            "Check your internet connection and dataset URL, "
            "or verify the model files."
        )

        st.stop()

if model_metrics is not None:

    st.success(
        "XGBoost model trained and saved successfully."
    )

else:

    st.success(
        "Saved XGBoost model loaded successfully."
    )


# ==========================================================
# 7. SESSION STATE
# ==========================================================

if "history" not in st.session_state:

    st.session_state.history = []

if "page" not in st.session_state:

    st.session_state.page = "Dashboard"

if "last_result" not in st.session_state:

    st.session_state.last_result = None


# ==========================================================
# 8. SIDEBAR NAVIGATION
# ==========================================================

st.sidebar.title("🏭 Smart Manufacturing")

st.sidebar.caption("AI Predictive Maintenance System")

pages = [
    "Dashboard",
    "Machine Monitoring",
    "Prediction",
    "History",
    "Model Performance",
    "About Project"
]

selected_page = st.sidebar.radio(
    "Navigation",
    pages,
    index=pages.index(st.session_state.page),
    key="navigation"
)

st.session_state.page = selected_page

st.sidebar.divider()

st.sidebar.success("XGBoost Model: Ready")

st.sidebar.caption(
    "Dataset: AI4I 2020 Predictive Maintenance"
)


# ==========================================================
# 9. MACHINE SENSOR INPUTS
# ==========================================================

def machine_input_form():

    st.subheader("⚙️ Machine Sensor Configuration")

    col1, col2 = st.columns(2)

    with col1:

        machine_type = st.selectbox(
            "Machine Type",
            ["L", "M", "H"],
            key="machine_type"
        )

        air_temp = st.number_input(
            "Air Temperature (K)",
            290.0,
            310.0,
            300.0,
            step=0.1
        )

        process_temp = st.number_input(
            "Process Temperature (K)",
            290.0,
            320.0,
            310.0,
            step=0.1
        )

    with col2:

        rotational_speed = st.number_input(
            "Rotational Speed (rpm)",
            1000,
            3000,
            1500,
            step=10
        )

        torque = st.number_input(
            "Torque (Nm)",
            0.0,
            100.0,
            40.0,
            step=0.5
        )

        tool_wear = st.number_input(
            "Tool Wear (min)",
            0,
            300,
            100,
            step=1
        )

    return {
        "Type": machine_type,
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": rotational_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear
    }


def prepare_input(values):

    data = pd.DataFrame([values])

    data = create_features(data)

    return data[FEATURE_COLUMNS]


# ==========================================================
# 10. MAKE PREDICTION
# ==========================================================

def predict_machine(values):

    input_data = prepare_input(values)

    processed_data = preprocessor.transform(
        input_data
    )

    prediction = int(
        model.predict(processed_data)[0]
    )

    probabilities = model.predict_proba(
        processed_data
    )[0]

    normal_probability = float(probabilities[0] * 100)

    failure_probability = float(probabilities[1] * 100)

    status = (
        "Machine Failure Predicted"
        if prediction == 1
        else "Normal Machine Operation"
    )

    result = {
        "Time": pd.Timestamp.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "Machine Type": values["Type"],
        "Air Temperature (K)": values["Air temperature [K]"],
        "Process Temperature (K)": values["Process temperature [K]"],
        "Rotational Speed (rpm)": values["Rotational speed [rpm]"],
        "Torque (Nm)": values["Torque [Nm]"],
        "Tool Wear (min)": values["Tool wear [min]"],
        "Prediction": status,
        "Failure Probability (%)": round(
            failure_probability, 2
        ),
        "Normal Probability (%)": round(
            normal_probability, 2
        )
    }

    return prediction, result


# ==========================================================
# 11. DASHBOARD PAGE
# ==========================================================

if st.session_state.page == "Dashboard":

    st.markdown(
        '<p class="dashboard-title">📊 Interactive Dashboard</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="dashboard-subtitle">'
        'Machine analytics and predictive maintenance overview'
        '</p>',
        unsafe_allow_html=True
    )

    history_df = pd.DataFrame(
        st.session_state.history
    )

    total = len(history_df)

    if total > 0:

        failures = int(
            (
                history_df["Prediction"]
                == "Machine Failure Predicted"
            ).sum()
        )

        normal = total - failures

        avg_failure_probability = history_df[
            "Failure Probability (%)"
        ].mean()

    else:

        failures = 0
        normal = 0
        avg_failure_probability = 0.0

    # KPI CARDS

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Predictions",
        total
    )

    c2.metric(
        "Failure Predictions",
        failures
    )

    c3.metric(
        "Normal Predictions",
        normal
    )

    c4.metric(
        "Average Failure Probability",
        f"{avg_failure_probability:.2f}%"
    )

    st.divider()

    # DASHBOARD FILTER

    st.subheader("🔎 Dashboard Filters")

    if total > 0:

        available_types = sorted(
            history_df["Machine Type"].unique().tolist()
        )

        selected_types = st.multiselect(
            "Filter by Machine Type",
            available_types,
            default=available_types
        )

        filtered_df = history_df[
            history_df["Machine Type"].isin(selected_types)
        ]

    else:

        filtered_df = history_df

        st.info(
            "Make predictions in the Prediction page "
            "to populate dashboard analytics."
        )

    # CHARTS

    chart1, chart2 = st.columns(2)

    with chart1:

        st.subheader("Machine Status Distribution")

        if len(filtered_df) > 0:

            counts = filtered_df["Prediction"].value_counts()

            fig = px.pie(
                values=counts.values,
                names=counts.index,
                hole=0.5,
                title="Normal vs Failure Predictions"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.info("No prediction data available.")

    with chart2:

        st.subheader("Failure Probability")

        if len(filtered_df) > 0:

            fig = px.histogram(
                filtered_df,
                x="Failure Probability (%)",
                nbins=10,
                title="Distribution of Failure Probability"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.info("No probability data available.")

    st.divider()

    # TREND CHART

    st.subheader("📈 Failure Probability Trend")

    if len(filtered_df) > 0:

        fig = px.line(
            filtered_df,
            x="Time",
            y="Failure Probability (%)",
            color="Machine Type",
            markers=True
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.info("Prediction trends will appear here.")

    # RECENT PREDICTIONS

    st.subheader("🕒 Recent Predictions")

    if len(filtered_df) > 0:

        st.dataframe(
            filtered_df.tail(10).iloc[::-1],
            use_container_width=True,
            hide_index=True
        )


# ==========================================================
# 12. MACHINE MONITORING PAGE
# ==========================================================

elif st.session_state.page == "Machine Monitoring":

    st.title("⚙️ Machine Monitoring")

    values = machine_input_form()

    air = values["Air temperature [K]"]
    process = values["Process temperature [K]"]
    speed = values["Rotational speed [rpm]"]
    torque_value = values["Torque [Nm]"]
    wear = values["Tool wear [min]"]

    temp_difference = process - air

    power = (
        2 * np.pi * speed * torque_value / 60
    )

    if wear <= 50:
        wear_category = "Low"
    elif wear <= 150:
        wear_category = "Medium"
    elif wear <= 250:
        wear_category = "High"
    else:
        wear_category = "Very High"

    st.divider()

    st.subheader("Live Input Sensor Overview")

    m1, m2, m3 = st.columns(3)

    m1.metric("Air Temperature", f"{air:.2f} K")
    m2.metric("Process Temperature", f"{process:.2f} K")
    m3.metric("Temperature Difference", f"{temp_difference:.2f} K")

    m4, m5, m6 = st.columns(3)

    m4.metric("Rotational Speed", f"{speed} rpm")
    m5.metric("Torque", f"{torque_value:.2f} Nm")
    m6.metric("Tool Wear", f"{wear} min")

    m7, m8 = st.columns(2)

    m7.metric("Calculated Power", f"{power:.2f} W")
    m8.metric("Tool Wear Category", wear_category)

    sensor_df = pd.DataFrame({
        "Sensor": [
            "Air Temperature",
            "Process Temperature",
            "Rotational Speed",
            "Torque",
            "Tool Wear"
        ],
        "Value": [
            air,
            process,
            speed,
            torque_value,
            wear
        ]
    })

    st.subheader("Sensor Visualization")

    fig = px.bar(
        sensor_df,
        x="Sensor",
        y="Value",
        color="Sensor",
        title="Machine Sensor Measurements"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    with st.expander("View Engineered Features"):

        engineered = prepare_input(values)

        st.dataframe(
            engineered,
            use_container_width=True
        )


# ==========================================================
# 13. PREDICTION PAGE
# ==========================================================

elif st.session_state.page == "Prediction":

    st.title("🤖 Machine Failure Prediction")

    st.write(
        "Enter machine sensor readings to predict the "
        "possibility of machine failure."
    )

    values = machine_input_form()

    st.divider()

    if st.button(
        "🚀 Predict Machine Failure",
        type="primary",
        use_container_width=True
    ):

        try:

            prediction, result = predict_machine(values)

            st.session_state.history.append(result)

            st.session_state.last_result = result

            st.subheader("Prediction Result")

            if prediction == 1:

                st.error("⚠️ Machine Failure Predicted")

                st.warning(
                    "The model predicts a possible machine failure. "
                    "Please inspect the machine."
                )

            else:

                st.success("✅ Normal Machine Operation")

                st.info(
                    "The model predicts normal operation "
                    "for the supplied sensor values."
                )

            p1, p2 = st.columns(2)

            p1.metric(
                "Failure Probability",
                f"{result['Failure Probability (%)']:.2f}%"
            )

            p2.metric(
                "Normal Probability",
                f"{result['Normal Probability (%)']:.2f}%"
            )

            probability_df = pd.DataFrame({
                "Condition": ["Normal", "Failure"],
                "Probability": [
                    result["Normal Probability (%)"],
                    result["Failure Probability (%)"]
                ]
            })

            fig = px.bar(
                probability_df,
                x="Condition",
                y="Probability",
                color="Condition",
                text_auto=".2f",
                title="Prediction Probability"
            )

            fig.update_layout(
                yaxis_range=[0, 100]
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        except Exception as error:

            st.error(
                f"Prediction failed: {error}"
            )


# ==========================================================
# 14. HISTORY PAGE
# ==========================================================

elif st.session_state.page == "History":

    st.title("📋 Prediction History")

    if st.session_state.history:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        st.metric(
            "Total Recorded Predictions",
            len(history_df)
        )

        st.dataframe(
            history_df.sort_values(
                "Time",
                ascending=False
            ),
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Prediction Analytics")

        fig = px.scatter(
            history_df,
            x="Time",
            y="Failure Probability (%)",
            color="Prediction",
            size="Tool Wear (min)",
            hover_data=[
                "Machine Type",
                "Rotational Speed (rpm)",
                "Torque (Nm)"
            ]
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        csv_data = history_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "📥 Download Prediction History",
            data=csv_data,
            file_name="machine_prediction_history.csv",
            mime="text/csv",
            use_container_width=True
        )

        if st.button(
            "🗑️ Clear Prediction History"
        ):

            st.session_state.history = []
            st.session_state.last_result = None

            st.rerun()

    else:

        st.info(
            "No prediction records available. "
            "Go to Prediction and run your first prediction."
        )


# ==========================================================
# 15. MODEL PERFORMANCE PAGE
# ==========================================================

elif st.session_state.page == "Model Performance":

    st.title("📈 Model Performance")

    if model_metrics is not None:

        st.info(
            "These are the metrics from the model trained "
            "during this application session."
        )

        cols = st.columns(4)

        for col, (name, value) in zip(
            cols, model_metrics.items()
        ):

            col.metric(
                name,
                f"{value:.2%}"
            )

        metric_df = pd.DataFrame({
            "Metric": list(model_metrics.keys()),
            "Score": list(model_metrics.values())
        })

        fig = px.bar(
            metric_df,
            x="Metric",
            y="Score",
            text_auto=".2%",
            title="XGBoost Evaluation Metrics"
        )

        fig.update_layout(
            yaxis_range=[0, 1]
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.info(
            "The saved model was loaded. Evaluation metrics "
            "were not recalculated in this session."
        )


# ==========================================================
# 16. ABOUT PROJECT PAGE
# ==========================================================

elif st.session_state.page == "About":

    st.title("ℹ️ About Smart Manufacturing AI")

    st.markdown("""
    ### Project Overview

    Smart Manufacturing AI is a predictive maintenance
    application designed to estimate machine failure risk
    using machine operating parameters.

    ### Dataset

    **AI4I 2020 Predictive Maintenance Dataset**

    - 10,000 records
    - 14 original columns
    - Synthetic industrial machine data
    - Target: Machine Failure

    ### Technologies

    - Python
    - Pandas and NumPy
    - Scikit-learn
    - XGBoost
    - Streamlit
    - Plotly

    ### Machine Learning Workflow

    1. Dataset Collection
    2. Data Understanding
    3. Data Cleaning
    4. Exploratory Data Analysis
    5. Feature Engineering
    6. Data Preprocessing
    7. Model Training
    8. Model Evaluation
    9. Machine Failure Prediction
    10. Interactive Dashboard
    """)

    st.markdown(
        "[AI4I 2020 Dataset – UCI Repository]"
        "(https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)"
    )

    st.warning(
        "This is an educational demonstration using synthetic data. "
        "It is not a live industrial monitoring system or a "
        "certified machine safety solution."
    )


# ==========================================================
# 17. FOOTER
# ==========================================================

st.divider()

st.caption(
    "Smart Manufacturing AI | Machine Monitoring and "
    "Failure Prediction | Python + XGBoost + Streamlit"
)
