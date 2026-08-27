# 🖥️ Streamlit Deployment Dashboard — US Apartment Rent Predictor

This directory contains the production-ready Streamlit web application and its automated MLflow training pipeline for predicting monthly US apartment rent.

---

## 📁 Artifacts & Files

| File | Type | Description |
| :--- | :--- | :--- |
| **[`app.py`](file:///Users/zeo/Downloads/RDS-Data_Science_Assignment/streamlit/app.py)** | Application | Interactive web interface with MLflow model registry loading, multi-model consensus, dynamic geolocation autofill, and confidence intervals |
| **[`train_model.py`](file:///Users/zeo/Downloads/RDS-Data_Science_Assignment/streamlit/train_model.py)** | Pipeline | Trains all 4 ML models, standardizes features, logs all metrics, models, and metadata to MLflow Registry |
| **`mlflow_run_map.joblib`** | Metadata | Active mapping between Streamlit UI model options and MLflow Model Run IDs |
| **`scaler.joblib`** | Preprocessor | Fitted `StandardScaler` for continuous feature normalization |
| **`num_cols.joblib`** | Metadata | List of continuous feature column names |
| **`model_columns.joblib`** | Metadata | Complete one-hot feature vector column alignment schema |
| **`state_geo.joblib`** | Metadata | Per-state median latitude / longitude lookup table |
| **`city_geo.joblib`** | Metadata | Per-city metadata (state mapping, GPS coordinates, historical median price) |
| **`model_metrics.joblib`** | Metadata | Serialized evaluation metrics for UI consensus comparison |
| **[`requirements.txt`](file:///Users/zeo/Downloads/RDS-Data_Science_Assignment/streamlit/requirements.txt)** | Dependencies | Python dependencies required to run the web app and training pipeline |

> **Note on Model Storage**: All 4 trained machine learning model binaries (`ApartmentRent_hist_gradient_boosting`, `ApartmentRent_random_forest`, `ApartmentRent_decision_tree`, `ApartmentRent_linear`) are stored and managed directly in **MLflow** (`../mlflow.db` & `../mlruns/`), eliminating bloated binary files in git.

---

## ⚡ Quickstart

### 1. Launch the Streamlit App
```bash
# From the project root directory:
streamlit run streamlit/app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 🔬 Viewing Experiments in MLflow UI

Launch the MLflow web interface to inspect experiments, model registry, metrics, and logged artifacts:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Open **`http://localhost:5000`** in your browser.

---

## 🔄 Retraining Models & Updating MLflow Registry

To retrain all 4 models and update the MLflow registry:

```bash
# Run the training script (requires apartments_for_rent_classified_100K.csv in the project root)
python streamlit/train_model.py
```
