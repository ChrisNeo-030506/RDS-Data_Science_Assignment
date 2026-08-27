# 🖥️ Streamlit Deployment Prototype — US Apartment Rent Predictor

This directory contains the production-ready Streamlit web application and its automated training pipeline for predicting monthly US apartment rent.

---

## 📁 Artifacts & Files

| File | Type | Description |
| :--- | :--- | :--- |
| **[`app.py`](file:///Users/zeo/Downloads/RDS-Data_Science_Assignment/streamlit/app.py)** | Application | Interactive web interface with multi-model consensus, dynamic geolocation autofill, and confidence intervals |
| **[`train_model.py`](file:///Users/zeo/Downloads/RDS-Data_Science_Assignment/streamlit/train_model.py)** | Pipeline | Trains all 4 ML models, standardizes features, logs runs to MLflow, and exports `.joblib` binaries |
| **`rent_model.joblib`** | Model | Default production model artifact (Hist Gradient Boosting) |
| **`model_hist_gradient_boosting.joblib`** | Model | Tuned Hist Gradient Boosting model binary (~7.2 MB) |
| **`model_random_forest.joblib`** | Model | 100-Tree Random Forest model binary (~140 MB) |
| **`model_decision_tree.joblib`** | Model | Tuned Decision Tree model binary (~0.5 MB) |
| **`model_linear.joblib`** | Model | Baseline Linear Regression model binary (< 3 KB) |
| **`model_metrics.joblib`** | Metadata | Serialized test holdout evaluation metrics for UI consensus comparison |
| **`scaler.joblib`** | Preprocessor | Fitted `StandardScaler` for continuous feature normalization |
| **`num_cols.joblib`** | Metadata | List of continuous feature column names |
| **`model_columns.joblib`** | Metadata | Complete one-hot feature vector column alignment schema |
| **`state_geo.joblib`** | Metadata | Per-state median latitude / longitude lookup table |
| **`city_geo.joblib`** | Metadata | Per-city metadata (state mapping, GPS coordinates, historical median price) |
| **[`requirements.txt`](file:///Users/zeo/Downloads/RDS-Data_Science_Assignment/streamlit/requirements.txt)** | Dependencies | Python dependencies required to run the web app and training pipeline |

---

## ⚡ Quickstart

### 1. Set Up Virtual Environment & Dependencies
```bash
# From the repository root:
python3 -m venv .venv
source .venv/bin/activate       # macOS / Linux
# or: .venv\Scripts\activate    # Windows

pip install -r streamlit/requirements.txt
```

### 2. Launch the Streamlit App
```bash
# Navigate to this folder
cd streamlit

# Run the app
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 🔄 Retraining Models & Updating Artifacts

If you modify feature engineering, hyperparameters, or encounter a scikit-learn version mismatch:

```bash
# Run the training script (requires apartments_for_rent_classified_100K.csv in the parent directory)
python train_model.py
```

This will retrain all 4 models, log runs to SQLite-backed MLflow, and regenerate all `.joblib` files.
