# Smart Manufacturing AI – Machine Monitoring and Failure Prediction

An end-to-end Machine Learning application designed to analyze industrial machine sensor data and predict potential machine failures. The project uses the AI4I 2020 Predictive Maintenance Dataset and an XGBoost classification model, presented through an interactive Streamlit dashboard.

## Live Application

**Streamlit Application:** [View Smart Manufacturing AI](https://smart-manufacturing-ai-nzdjnlfuvg4tfjelfjrfjh.streamlit.app)

**GitHub Repository:** [Smart Manufacturing AI](https://github.com/ajayvasudevancbe-Datascience/Smart-Manufacturing-AI)

## Project Overview

Unexpected equipment failures can result in production interruptions, increased maintenance expenses, and operational inefficiencies.

Smart Manufacturing AI demonstrates a predictive maintenance approach using machine sensor measurements and Machine Learning techniques to estimate the possibility of equipment failure.

The application enables users to analyze machine parameters, generate predictions, evaluate model performance, and explore uploaded datasets through an interactive dashboard.

## Project Objectives

- Analyze industrial machine sensor data.
- Perform data cleaning and preprocessing.
- Conduct exploratory data analysis.
- Develop meaningful features from sensor measurements.
- Implement an XGBoost classification model.
- Evaluate model performance using classification metrics.
- Develop an interactive monitoring and prediction dashboard.
- Demonstrate a predictive maintenance workflow.

## Dataset Information

**Dataset Name:** AI4I 2020 Predictive Maintenance Dataset

**Source:** UCI Machine Learning Repository

**Dataset URL:** https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

**CSV Download:** https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv

| Attribute | Description |
|---|---|
| Number of Records | 10,000 |
| Original Features | 14 |
| Target Variable | Machine failure |
| Problem Type | Binary Classification |
| Dataset Type | Synthetic industrial sensor data |

### Important Features

- Machine Type
- Air Temperature
- Process Temperature
- Rotational Speed
- Torque
- Tool Wear
- Machine Failure

Failure-mode columns are excluded from model input to prevent target leakage.

## Technologies and Tools

| Technology | Application |
|---|---|
| Python | Programming and model development |
| Pandas | Data manipulation |
| NumPy | Numerical computation |
| Scikit-learn | Preprocessing and model evaluation |
| XGBoost | Classification model |
| Plotly | Interactive data visualization |
| Streamlit | Web application development |
| Joblib | Model serialization |

## Machine Learning Workflow

1. Dataset Collection
2. Data Understanding
3. Data Cleaning
4. Exploratory Data Analysis
5. Feature Engineering
6. Data Preprocessing
7. Train-Test Splitting
8. Model Development
9. Model Evaluation
10. Machine Failure Prediction
11. Interactive Dashboard Development
12. Application Deployment

## Feature Engineering

Additional features are generated from the original machine sensor measurements.

### Temperature Difference

Calculated using process temperature and air temperature.

### Mechanical Power

Estimated using rotational speed and torque.

### Tool Wear Category

Tool wear measurements are categorized into four groups:

- Low
- Medium
- High
- Very High

## Machine Learning Model

**Algorithm:** XGBoost Classifier

**Prediction Classes:**

- 0: Normal Operation
- 1: Machine Failure

The dataset is divided into 80% training data and 20% testing data using stratified sampling.

Numerical features are processed using median imputation and standardization. Categorical features are processed using most-frequent imputation and One-Hot Encoding.

Class weighting is applied to help address the imbalance between normal and failure records.

## Model Evaluation

The trained model is evaluated using the following classification metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

Evaluation metrics are calculated using the held-out test dataset and displayed in the Streamlit application.

Actual evaluation scores should be added here after running the final model.

## Application Features

### Dashboard

Displays prediction statistics, including total predictions, normal predictions, failure predictions, and average failure probability. Includes interactive charts for prediction analysis.

### Machine Monitoring

Allows users to enter machine sensor measurements, calculate engineered features, and visualize sensor readings.

### Machine Failure Prediction

Accepts machine parameters and generates normal or failure predictions with estimated failure probability.

### Prediction History

Maintains prediction records during the current application session. Supports data visualization and CSV export.

### Model Performance

Displays model evaluation metrics, confusion matrix, and classification report.

### Sample Data Analysis

Supports CSV file uploads and provides dataset exploration features, including:

- Dataset preview
- Row and column information
- Missing-value analysis
- Column data types
- Unique-value counts
- Numerical statistics
- Interactive visualizations
- Machine failure distribution
- CSV download

Uploaded sample datasets are used for exploration and do not automatically retrain the prediction model.

### About Project

Provides project objectives, dataset information, technologies, and the Machine Learning workflow.

## Project Structure

```text
Smart-Manufacturing-AI/
|
|-- app.py
|-- requirements.txt
|-- README.md
|
|-- smart_manufacturing_model/
    |-- xgb_model.pkl
    |-- preprocessor.pkl
    |-- model_metrics.pkl
```

Model artifacts are generated automatically when the application trains the model successfully.

## Installation and Execution

### 1. Clone the Repository

```bash
git clone https://github.com/ajayvasudevancbe-Datascience/Smart-Manufacturing-AI.git
```

### 2. Navigate to the Project Directory

```bash
cd Smart-Manufacturing-AI
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

### 5. Access the Application

Open the local URL provided by Streamlit:

```text
http://localhost:8501
```

The application downloads the dataset and trains the model automatically when saved model files are unavailable.

## Deployment

The application can be deployed using Streamlit Community Cloud.

Deployment Platform: https://share.streamlit.io/

Deployment Process:

1. Push the project files to GitHub.
2. Connect the GitHub repository to Streamlit Community Cloud.
3. Select `app.py` as the main application file.
4. Deploy the application.
5. Add the generated public application URL to the Live Application section above.

## Future Enhancements

- Real-time industrial sensor integration.
- IoT-based machine monitoring.
- Live sensor data visualization.
- Machine Learning and Deep Learning model comparison.
- Explainable AI for prediction interpretation.
- Automated maintenance notifications.
- Cloud-based model deployment.

## Limitations

- The AI4I dataset contains synthetic benchmark data rather than actual factory sensor recordings.
- Predictions are estimates based on the training dataset.
- The current application uses manually entered sensor values rather than live industrial sensor connections.
- The application is a demonstration of predictive maintenance and is not a validated industrial safety system.

## Author

**Ajay Vasudevan**

B.Sc. Data Science

GitHub: https://github.com/ajayvasudevancbe-Datascience

LinkedIn: https://www.linkedin.com/in/ajay-vasudevan-8800953a1/



