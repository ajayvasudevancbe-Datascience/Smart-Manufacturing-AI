
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from xgboost import XGBClassifier

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Smart Manufacturing AI",
    page_icon="🏭",
    layout="wide"
)

st.title("🏭 Smart Manufacturing AI")
st.caption("Machine Monitoring and Predictive Failure Detection")


# ============================================================
# 2. CONFIGURATION
# ============================================================

DATA_URL = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/00601/ai4i2020.csv"
)

MODEL_DIR = "smart_manufacturing_model"

MODEL_PATH = os.path.join(MODEL_DIR, "xgb_model.pkl")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "model_metrics.pkl")

NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Temperature Difference [K]",
    "Power [W]"
]

CATEGORICAL_FEATURES = [
    "Type",
    "Tool Wear Category"
]

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES


# ============================================================
# 3. SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None


# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================

def create_features(data):

    df = data.copy()

    required_columns = [
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
        )

    df["Temperature Difference [K]"] = (
        df["Process temperature [K]"]
        - df["Air temperature [K]"]
    )

    df["Power [W]"] = (
        2 * np.pi
        * df["Rotational speed [rpm]"]
        * df["Torque [Nm]"] / 60
    )

    df["Tool Wear Category"] = pd.cut(
        df["Tool wear [min]"],
        bins=[-1, 50, 150, 250, np.inf],
        labels=["Low", "Medium", "High", "Very High"]
    )

    return df


# ============================================================
# 5. MODEL EVALUATION FUNCTION
# ============================================================

def evaluate_model(model, preprocessor, X_test, y_test):

    X_test_processed = preprocessor.transform(X_test)

    predictions = model.predict(X_test_processed)

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),

        "Precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "Recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "F1 Score": f1_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "Confusion Matrix": confusion_matrix(
            y_test,
            predictions
        ),

        "Classification Report": classification_report(
            y_test,
            predictions,
            zero_division=0,
            output_dict=True
        )
    }

    return metrics


# ============================================================
# 6. MODEL TRAINING AND LOADING
# ============================================================

@st.cache_resource
def load_or_train_model():

    os.makedirs(MODEL_DIR, exist_ok=True)

    # Download dataset
    df = pd.read_csv(DATA_URL)

    df = create_features(df)

    X = df[FEATURE_COLUMNS].copy()
    y = df["Machine failure"].astype(int)

    # Fixed train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # --------------------------------------------------------
    # LOAD EXISTING MODEL
    # --------------------------------------------------------

    if (
        os.path.exists(MODEL_PATH)
        and os.path.exists(PREPROCESSOR_PATH)
    ):

        model = joblib.load(MODEL_PATH)

        preprocessor = joblib.load(PREPROCESSOR_PATH)

        # Load saved metrics when available
        if os.path.exists(METRICS_PATH):

            metrics = joblib.load(METRICS_PATH)

        else:

            # Calculate metrics for saved model
            metrics = evaluate_model(
                model,
                preprocessor,
                X_test,
                y_test
            )

            joblib.dump(metrics, METRICS_PATH)

        return (
            model,
            preprocessor,
            metrics,
            "Saved model loaded and evaluated successfully"
        )

    # --------------------------------------------------------
    # TRAIN NEW MODEL
    # --------------------------------------------------------

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ])

    preprocessor = ColumnTransformer([
        (
            "num",
            numeric_pipeline,
            NUMERIC_FEATURES
        ),
        (
            "cat",
            categorical_pipeline,
            CATEGORICAL_FEATURES
        )
    ])

    X_train_processed = preprocessor.fit_transform(X_train)

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

    # Evaluate new model
    metrics = evaluate_model(
        model,
        preprocessor,
        X_test,
        y_test
    )

    # Save all files
    joblib.dump(model, MODEL_PATH)

    joblib.dump(
        preprocessor,
        PREPROCESSOR_PATH
    )

    joblib.dump(
        metrics,
        METRICS_PATH
    )

    return (
        model,
        preprocessor,
        metrics,
        "New model trained and evaluated successfully"
    )


# ============================================================
# 7. LOAD MODEL
# ============================================================

try:

    with st.spinner("Loading and evaluating AI model..."):

        model, preprocessor, model_metrics, model_status = (
            load_or_train_model()
        )

    st.sidebar.success(model_status)

except Exception as error:

    st.error(f"Model loading error: {error}")

    st.info(
        "Check your internet connection and installed requirements."
    )

    st.stop()


# ============================================================
# 8. SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("🏭 Navigation")

pages = [
    "Dashboard",
    "Machine Monitoring",
    "Prediction",
    "History",
    "Model Performance",
    "Sample Data",
    "About Project"
]

selected_page = st.sidebar.radio(
    "Select Page",
    pages
)

st.sidebar.divider()

st.sidebar.caption(
    "Smart Manufacturing AI"
)


# ============================================================
# 9. PREDICTION FUNCTION
# ============================================================

def predict_failure(input_data):

    input_df = pd.DataFrame([input_data])

    processed = preprocessor.transform(input_df)

    prediction = int(
        model.predict(processed)[0]
    )

    probabilities = model.predict_proba(processed)[0]

    failure_probability = float(
        probabilities[1] * 100
    )

    return prediction, failure_probability


# ============================================================
# PAGE 1: DASHBOARD
# ============================================================

if selected_page == "Dashboard":

    st.header("📊 Dashboard")

    history_df = pd.DataFrame(
        st.session_state.history
    )

    total = len(history_df)

    failures = (
        int((history_df["Prediction"] == "Failure").sum())
        if total else 0
    )

    normal = total - failures

    average_probability = (
        history_df["Failure Probability (%)"].mean()
        if total else 0
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Predictions", total)

    col2.metric("Failure Predictions", failures)

    col3.metric("Normal Predictions", normal)

    col4.metric(
        "Average Failure Probability",
        f"{average_probability:.2f}%"
    )

    st.divider()

    if total > 0:

        left, right = st.columns(2)

        with left:

            st.subheader("Prediction Distribution")

            distribution = (
                history_df["Prediction"].value_counts()
            )

            fig = px.pie(
                values=distribution.values,
                names=distribution.index,
                title="Normal vs Failure"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with right:

            st.subheader("Failure Probability")

            fig = px.histogram(
                history_df,
                x="Failure Probability (%)",
                nbins=15,
                title="Prediction Probability Distribution"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.subheader("Recent Predictions")

        st.dataframe(
            history_df.tail(10).iloc[::-1],
            use_container_width=True
        )

    else:

        st.info(
            "No prediction history available. "
            "Make your first prediction."
        )


# ============================================================
# PAGE 2: MACHINE MONITORING
# ============================================================

elif selected_page == "Machine Monitoring":

    st.header("⚙️ Machine Monitoring")

    st.write(
        "Enter machine sensor readings to calculate monitoring features."
    )

    col1, col2 = st.columns(2)

    with col1:

        machine_type = st.selectbox(
            "Machine Type",
            ["L", "M", "H"]
        )

        air_temp = st.number_input(
            "Air Temperature (K)",
            min_value=250.0,
            max_value=350.0,
            value=298.0
        )

        process_temp = st.number_input(
            "Process Temperature (K)",
            min_value=250.0,
            max_value=400.0,
            value=308.0
        )

    with col2:

        speed = st.number_input(
            "Rotational Speed (rpm)",
            min_value=500,
            max_value=4000,
            value=1500
        )

        torque = st.number_input(
            "Torque (Nm)",
            min_value=1.0,
            max_value=100.0,
            value=40.0
        )

        wear = st.number_input(
            "Tool Wear (min)",
            min_value=0,
            max_value=300,
            value=100
        )

    temperature_difference = (
        process_temp - air_temp
    )

    power = (
        2 * np.pi * speed * torque / 60
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

    st.subheader("Calculated Sensor Features")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Temperature Difference",
        f"{temperature_difference:.2f} K"
    )

    c2.metric(
        "Power",
        f"{power:.2f} W"
    )

    c3.metric(
        "Tool Wear Category",
        wear_category
    )

    sensor_df = pd.DataFrame({
        "Sensor": [
            "Air Temperature",
            "Process Temperature",
            "Rotational Speed",
            "Torque",
            "Tool Wear"
        ],
        "Value": [
            air_temp,
            process_temp,
            speed,
            torque,
            wear
        ]
    })

    st.subheader("Sensor Readings")

    fig = px.bar(
        sensor_df,
        x="Sensor",
        y="Value",
        title="Machine Sensor Values"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.caption(
        "These are manually entered readings, not live factory sensor data."
    )


# ============================================================
# PAGE 3: PREDICTION
# ============================================================

elif selected_page == "Prediction":

    st.header("🤖 Machine Failure Prediction")

    st.write(
        "Enter machine sensor values to predict possible machine failure."
    )

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:

            machine_type = st.selectbox(
                "Machine Type",
                ["L", "M", "H"]
            )

            air_temp = st.number_input(
                "Air Temperature (K)",
                min_value=250.0,
                max_value=350.0,
                value=298.0
            )

            process_temp = st.number_input(
                "Process Temperature (K)",
                min_value=250.0,
                max_value=400.0,
                value=308.0
            )

            speed = st.number_input(
                "Rotational Speed (rpm)",
                min_value=500,
                max_value=4000,
                value=1500
            )

        with col2:

            torque = st.number_input(
                "Torque (Nm)",
                min_value=1.0,
                max_value=100.0,
                value=40.0
            )

            wear = st.number_input(
                "Tool Wear (min)",
                min_value=0,
                max_value=300,
                value=100
            )

        submitted = st.form_submit_button(
            "Predict Machine Failure",
            use_container_width=True
        )

    if submitted:

        if process_temp < air_temp:

            st.warning(
                "Process temperature is lower than air temperature. "
                "Please verify the readings."
            )

        else:

            input_data = {
                "Type": machine_type,

                "Air temperature [K]": air_temp,

                "Process temperature [K]": process_temp,

                "Rotational speed [rpm]": speed,

                "Torque [Nm]": torque,

                "Tool wear [min]": wear,

                "Temperature Difference [K]": (
                    process_temp - air_temp
                ),

                "Power [W]": (
                    2 * np.pi * speed * torque / 60
                ),

                "Tool Wear Category": wear_category if False else (
                    "Low" if wear <= 50
                    else "Medium" if wear <= 150
                    else "High" if wear <= 250
                    else "Very High"
                )
            }

            prediction, probability = predict_failure(
                input_data
            )

            result = (
                "Failure" if prediction == 1 else "Normal"
            )

            st.session_state.last_result = {
                "Prediction": result,
                "Failure Probability (%)": probability
            }

            st.session_state.history.append({
                "Time": pd.Timestamp.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

                "Machine Type": machine_type,

                "Air Temperature [K]": air_temp,

                "Process Temperature [K]": process_temp,

                "Rotational Speed [rpm]": speed,

                "Torque [Nm]": torque,

                "Tool Wear [min]": wear,

                "Prediction": result,

                "Failure Probability (%)": round(
                    probability, 2
                )
            })

    if st.session_state.last_result is not None:

        result = st.session_state.last_result["Prediction"]

        probability = st.session_state.last_result[
            "Failure Probability (%)"
        ]

        st.divider()

        st.subheader("Prediction Result")

        if result == "Failure":

            st.error("⚠️ Predicted Machine Failure")

        else:

            st.success("✅ Predicted Normal Operation")

        col1, col2 = st.columns(2)

        col1.metric("Prediction", result)

        col2.metric(
            "Failure Probability",
            f"{probability:.2f}%"
        )

        probability_df = pd.DataFrame({
            "Class": ["Normal", "Failure"],

            "Probability": [
                100 - probability,
                probability
            ]
        })

        fig = px.bar(
            probability_df,
            x="Class",
            y="Probability",
            text_auto=".2f",
            title="Prediction Probability"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.caption(
            "This is a model estimate, not a confirmed diagnosis "
            "of actual machine condition."
        )


# ============================================================
# PAGE 4: HISTORY
# ============================================================

elif selected_page == "History":

    st.header("📜 Prediction History")

    history_df = pd.DataFrame(
        st.session_state.history
    )

    if not history_df.empty:

        st.subheader("All Predictions")

        st.dataframe(
            history_df,
            use_container_width=True
        )

        st.subheader("Prediction Probability Analysis")

        fig = px.scatter(
            history_df,
            x="Time",
            y="Failure Probability (%)",
            color="Prediction",
            title="Failure Probability Over Time"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        csv_data = history_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download Prediction History CSV",
            data=csv_data,
            file_name="machine_prediction_history.csv",
            mime="text/csv"
        )

        if st.button("Clear Prediction History"):

            st.session_state.history = []

            st.session_state.last_result = None

            st.rerun()

    else:

        st.info("No prediction history available yet.")


# ============================================================
# PAGE 5: MODEL PERFORMANCE
# ============================================================

elif selected_page == "Model Performance":

    st.header("📈 Model Performance")

    st.write(
        "Evaluation results from the model's held-out test dataset."
    )

    if model_metrics is not None:

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Accuracy",
            f"{model_metrics['Accuracy'] * 100:.2f}%"
        )

        col2.metric(
            "Precision",
            f"{model_metrics['Precision'] * 100:.2f}%"
        )

        col3.metric(
            "Recall",
            f"{model_metrics['Recall'] * 100:.2f}%"
        )

        col4.metric(
            "F1 Score",
            f"{model_metrics['F1 Score'] * 100:.2f}%"
        )

        st.divider()

        st.subheader("Evaluation Metrics")

        metric_df = pd.DataFrame({
            "Metric": [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score"
            ],

            "Score": [
                model_metrics["Accuracy"],
                model_metrics["Precision"],
                model_metrics["Recall"],
                model_metrics["F1 Score"]
            ]
        })

        fig = px.bar(
            metric_df,
            x="Metric",
            y="Score",
            text=metric_df["Score"].map(
                lambda value: f"{value:.3f}"
            ),
            title="XGBoost Model Evaluation"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader("Confusion Matrix")

        cm = np.array(
            model_metrics["Confusion Matrix"]
        )

        fig = px.imshow(
            cm,
            text_auto=True,
            x=["Predicted Normal", "Predicted Failure"],
            y=["Actual Normal", "Actual Failure"],
            labels={
                "x": "Predicted",
                "y": "Actual",
                "color": "Count"
            },
            title="Confusion Matrix"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader("Classification Report")

        report = pd.DataFrame(
            model_metrics["Classification Report"]
        ).transpose()

        st.dataframe(
            report,
            use_container_width=True
        )

        st.success(
            "Evaluation metrics successfully loaded or calculated."
        )

    else:

        st.warning(
            "Evaluation metrics are not available."
        )


# ============================================================
# PAGE 6: SAMPLE DATA
# ============================================================

elif selected_page == "Sample Data":

    st.header("📂 Sample Data Analysis")

    st.write(
        "Upload a CSV file to explore its actual dataset values."
    )

    st.info(
        "Uploaded data is used for exploration only. "
        "It is not automatically used to retrain the model."
    )

    uploaded_file = st.file_uploader(
        "Upload Sample CSV Dataset",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            sample_df = pd.read_csv(uploaded_file)

            if sample_df.empty:

                st.warning(
                    "The uploaded CSV file contains no records."
                )

            else:

                st.success(
                    "CSV file uploaded successfully!"
                )

                # Dataset overview
                st.subheader("Dataset Overview")

                total_rows = sample_df.shape[0]

                total_columns = sample_df.shape[1]

                missing_values = int(
                    sample_df.isnull().sum().sum()
                )

                col1, col2, col3 = st.columns(3)

                col1.metric("Total Rows", total_rows)

                col2.metric("Total Columns", total_columns)

                col3.metric("Missing Values", missing_values)

                st.divider()

                # Display complete dataset
                st.subheader("Uploaded Dataset Values")

                st.dataframe(
                    sample_df,
                    use_container_width=True,
                    height=400
                )

                # Display sample records
                st.subheader("Sample Records")

                number_of_rows = st.slider(
                    "Select number of rows",
                    min_value=1,
                    max_value=min(total_rows, 100),
                    value=min(10, total_rows)
                )

                st.dataframe(
                    sample_df.head(number_of_rows),
                    use_container_width=True
                )

                st.divider()

                # Column information
                st.subheader("Column Information")

                column_info = pd.DataFrame({
                    "Column Name": sample_df.columns,

                    "Data Type": [
                        str(dtype)
                        for dtype in sample_df.dtypes
                    ],

                    "Missing Values": [
                        int(value)
                        for value in sample_df.isnull().sum()
                    ],

                    "Unique Values": [
                        int(sample_df[col].nunique(dropna=True))
                        for col in sample_df.columns
                    ]
                })

                st.dataframe(
                    column_info,
                    use_container_width=True
                )

                st.divider()

                # Numerical statistics
                st.subheader("Numerical Statistics")

                numeric_df = sample_df.select_dtypes(
                    include=np.number
                )

                if not numeric_df.empty:

                    st.dataframe(
                        numeric_df.describe().round(2),
                        use_container_width=True
                    )

                else:

                    st.info(
                        "No numerical columns found."
                    )

                st.divider()

                # Data visualization
                st.subheader("Data Visualization")

                available_columns = sample_df.columns.tolist()

                selected_column = st.selectbox(
                    "Select a column",
                    available_columns
                )

                chart_type = st.selectbox(
                    "Select chart type",
                    [
                        "Histogram",
                        "Box Plot",
                        "Bar Chart",
                        "Line Chart"
                    ]
                )

                if chart_type == "Histogram":

                    if pd.api.types.is_numeric_dtype(
                        sample_df[selected_column]
                    ):

                        fig = px.histogram(
                            sample_df,
                            x=selected_column,
                            title=f"Distribution of {selected_column}"
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True
                        )

                    else:

                        st.warning(
                            "Histogram requires a numerical column."
                        )

                elif chart_type == "Box Plot":

                    if pd.api.types.is_numeric_dtype(
                        sample_df[selected_column]
                    ):

                        fig = px.box(
                            sample_df,
                            y=selected_column,
                            title=f"Box Plot of {selected_column}"
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True
                        )

                    else:

                        st.warning(
                            "Box Plot requires a numerical column."
                        )

                elif chart_type == "Bar Chart":

                    counts = (
                        sample_df[selected_column]
                        .astype(str)
                        .value_counts()
                        .head(30)
                        .reset_index()
                    )

                    counts.columns = [
                        selected_column,
                        "Count"
                    ]

                    fig = px.bar(
                        counts,
                        x=selected_column,
                        y="Count",
                        title=f"Count of {selected_column}"
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                elif chart_type == "Line Chart":

                    if pd.api.types.is_numeric_dtype(
                        sample_df[selected_column]
                    ):

                        fig = px.line(
                            sample_df,
                            y=selected_column,
                            title=f"Line Chart of {selected_column}"
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True
                        )

                    else:

                        st.warning(
                            "Line Chart requires a numerical column."
                        )

                # Machine failure distribution
                if "Machine failure" in sample_df.columns:

                    st.divider()

                    st.subheader(
                        "Machine Failure Distribution"
                    )

                    failure_counts = (
                        sample_df["Machine failure"]
                        .value_counts(dropna=False)
                        .reset_index()
                    )

                    failure_counts.columns = [
                        "Machine Failure",
                        "Count"
                    ]

                    fig = px.pie(
                        failure_counts,
                        names="Machine Failure",
                        values="Count",
                        title="Normal vs Failure Distribution"
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                # Download uploaded data
                st.divider()

                csv_data = sample_df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    "Download Uploaded Dataset",
                    data=csv_data,
                    file_name="uploaded_sample_data.csv",
                    mime="text/csv"
                )

        except Exception as error:

            st.error(
                f"Unable to read CSV file: {error}"
            )

    else:

        st.info(
            "Upload a CSV file above to display its values."
        )

        st.markdown(
            "[Download AI4I Dataset CSV]"
            "(https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv)"
        )


# ============================================================
# PAGE 7: ABOUT PROJECT
# ============================================================

elif selected_page == "About Project":

    st.header("ℹ️ About Smart Manufacturing AI")

    st.write(
        """
        **Smart Manufacturing AI – Machine Monitoring and Failure
        Prediction** is a machine learning application that estimates
        the possibility of industrial machine failure using sensor data.
        """
    )

    st.subheader("Project Objectives")

    st.markdown("""
    - Analyze machine sensor data.
    - Perform data cleaning and feature engineering.
    - Develop a machine failure prediction model.
    - Evaluate model performance.
    - Visualize machine predictions.
    - Explore uploaded CSV datasets.
    """)

    st.subheader("Dataset Information")

    st.markdown("""
    - Dataset: AI4I 2020 Predictive Maintenance Dataset
    - Source: UCI Machine Learning Repository
    - Records: 10,000
    - Target: Machine failure
    - Dataset type: Synthetic predictive maintenance data
    """)

    st.markdown(
        "[Open UCI Dataset]"
        "(https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)"
    )

    st.subheader("Technologies Used")

    st.markdown("""
    - Python
    - Pandas and NumPy
    - Scikit-learn
    - XGBoost
    - Plotly
    - Streamlit
    - Joblib
    - Machine Learning
    - Data Visualization
    """)

    st.subheader("Machine Learning Workflow")

    st.markdown("""
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

    st.caption(
        "This project uses a synthetic benchmark dataset. "
        "It is a demonstration of predictive maintenance, not a "
        "validated industrial safety system."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Smart Manufacturing AI | Machine Monitoring and Failure Prediction"
)






