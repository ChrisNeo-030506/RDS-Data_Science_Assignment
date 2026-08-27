# 🖥️ Streamlit Deployment Dashboard — US Apartment Rent Predictor

This directory contains the production-ready Streamlit web application and its automated training pipeline for predicting monthly US apartment rent.

---

## 📁 Artifacts & Files

| File | Type | Description |
| :--- | :--- | :--- |
| **[`app.py`](file:///Users/zeo/Downloads/RDS-Data_Science_Assignment/streamlit/app.py)** | Application | Interactive web interface with multi-model consensus, dynamic geolocation autofill, and confidence intervals |
| **[`train_model.py`](file:///Users/zeo/Downloads/RDS-Data_Science_Assignment/streamlit/train_model.py)** | Pipeline | Automated script to clean data, engineer features, train all 4 ML models, and export artifacts |
| **`rent_model.joblib`** | Model | Production default model artifact (Hist Gradient Boosting) |
| **`model_hist_gradient_boosting.joblib`** | Model | Hist Gradient Boosting model binary (~3.2 MB) |
| **`model_random_forest.joblib.part*`** | Model | Full-depth 100-Tree Random Forest model, chunked for safe GitHub hosting (< 50MB per part) |
| **`model_decision_tree.joblib`** | Model | Tuned Decision Tree model binary (~150 KB) |
| **`model_linear.joblib`** | Model | Baseline Linear Regression model binary (~2 KB) |
| **`scaler.joblib`** | Preprocessor | Fitted `StandardScaler` for continuous feature normalization |
| **`num_cols.joblib`** | Metadata | List of continuous feature column names |
| **`model_columns.joblib`** | Metadata | Complete one-hot feature vector column alignment schema |
| **`state_geo.joblib`** | Metadata | Per-state median latitude / longitude lookup table |
| **`city_geo.joblib`** | Metadata | Per-city metadata (state mapping, GPS coordinates, historical median price) |
| **`model_metrics.joblib`** | Metadata | Serialized evaluation metrics for UI consensus comparison |
| **[`requirements.txt`](file:///Users/zeo/Downloads/RDS-Data_Science_Assignment/streamlit/requirements.txt)** | Dependencies | Python dependencies required to run the web app and training pipeline |

---

## ⚡ Quickstart

### 1. Launch the Streamlit App
```bash
# From the project root directory:
streamlit run streamlit/app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 🔄 Retraining Models & Updating Artifacts

To retrain all 4 models and update all deployment artifacts:

```bash
# Run the training script (requires apartments_for_rent_classified_100K.csv in the project root)
python streamlit/train_model.py
```
