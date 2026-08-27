"""
train_model.py
---------------
Trains all 4 machine learning models (Linear Baseline, Decision Tree, Random Forest, Hist Gradient Boosting)
with advanced feature engineering, target log normalization (log1p), and feature standardization (StandardScaler).

Logs all models, hyperparameters, evaluation metrics, and preprocessing artifacts to MLflow Registry & Experiments.
Exports fallback .joblib binaries for deployment portability:
    model_linear.joblib                 - Model 1: Linear Regression (Baseline)
    model_decision_tree.joblib          - Model 2: Decision Tree (Tuned)
    model_random_forest.joblib          - Model 3: Random Forest (100 Trees, Compressed)
    model_hist_gradient_boosting.joblib - Model 4: Hist Gradient Boosting (Tuned & Regularized)
    rent_model.joblib                   - Default deployed model
    model_metrics.joblib                - Comparative benchmark metrics for UI
    scaler.joblib                       - Trained StandardScaler
    num_cols.joblib                     - Numerical column names
    model_columns.joblib                - One-hot aligned feature names
    state_geo.joblib                    - Per-state median coordinates
    city_geo.joblib                     - Per-city metadata (state, coords, median price)
    mlflow_run_map.joblib               - MLflow Run IDs and model metadata
"""
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import mlflow
import mlflow.sklearn

HERE = os.path.dirname(os.path.abspath(__file__))
CSV  = os.path.join(HERE, "..", "apartments_for_rent_classified_100K.csv")
DB_PATH = os.path.join(HERE, "..", "mlflow.db")

print("Loading dataset...")
df = pd.read_csv(CSV, sep=";", encoding="cp1252", low_memory=False)

# ---------- Filter & Clean ----------
dc = df[df["price_type"] == "Monthly"].copy()
dc = dc.drop(columns=["id", "category", "title", "body", "currency", "price_display",
                      "price_type", "address", "source", "time", "pets_allowed"])

def iqr_bounds(series, k=3.0):
    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    IQR = Q3 - Q1
    return Q1 - k * IQR, Q3 + k * IQR

price_low, price_high = iqr_bounds(dc["price"], 3.0)
sqft_low, sqft_high   = iqr_bounds(dc["square_feet"], 3.0)

dc = dc[(dc["price"] >= price_low) & (dc["price"] <= price_high) &
        (dc["square_feet"] >= sqft_low) & (dc["square_feet"] <= sqft_high)]
dc = dc.dropna(subset=["amenities", "cityname", "state", "latitude", "longitude"])
dc["bathrooms"] = dc["bathrooms"].fillna(dc["bathrooms"].median())
dc["bedrooms"]  = dc["bedrooms"].fillna(dc["bedrooms"].median())
dc = dc.drop_duplicates(keep="first")

# ---------- Feature Engineering ----------
amenities_lower = dc["amenities"].str.lower()
key_amenities = ["pool", "gym", "dishwasher", "parking", "garage", "washer",
                 "dryer", "ac", "patio", "gated", "fireplace", "elevator"]
for item in key_amenities:
    dc[f"has_{item}"] = amenities_lower.apply(lambda x: 1 if item in x else 0)

dc["amenity_count"] = dc["amenities"].apply(lambda x: 0 if x == "none" else len(x.split(",")))
dc["sqft_per_bedroom"]  = dc["square_feet"] / (dc["bedrooms"] + 1)
dc["sqft_per_bathroom"] = dc["square_feet"] / (dc["bathrooms"] + 1)
dc["bed_bath_ratio"]    = dc["bedrooms"] / (dc["bathrooms"] + 0.5)

# Save rich city metadata for UI
city_grouped = dc.groupby(["cityname", "state"]).agg({
    "price": "median",
    "latitude": "median",
    "longitude": "median"
}).reset_index()

# ---------- Split First to Prevent Target Leakage ----------
train_df, test_df = train_test_split(dc, test_size=0.2, random_state=42)

overall_median = train_df["price"].median()
city_medians = train_df.groupby("cityname")["price"].median().to_dict()

train_df["city_median_price"] = train_df["cityname"].map(city_medians).fillna(overall_median)
test_df["city_median_price"]  = test_df["cityname"].map(city_medians).fillna(overall_median)

# ---------- Encode ----------
drop_cols = ["cityname", "amenities"]
train_dm = train_df.drop(columns=drop_cols)
test_dm  = test_df.drop(columns=drop_cols)

train_dm = pd.get_dummies(train_dm, columns=["fee", "has_photo", "state"], drop_first=True)
test_dm  = pd.get_dummies(test_dm, columns=["fee", "has_photo", "state"], drop_first=True)
train_dm, test_dm = train_dm.align(test_dm, join="left", axis=1, fill_value=0)

X_train = train_dm.drop(columns=["price"])
y_train_orig = train_dm["price"]
y_train_log  = np.log1p(y_train_orig)

X_test = test_dm.drop(columns=["price"])
y_test_orig = test_dm["price"]
y_test_log  = np.log1p(y_test_orig)

# ---------- Normalization (StandardScaler) ----------
num_cols = ["square_feet", "bedrooms", "bathrooms", "amenity_count", "latitude", "longitude",
            "sqft_per_bedroom", "sqft_per_bathroom", "bed_bath_ratio", "city_median_price"]

scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled  = X_test.copy()
X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test_scaled[num_cols]  = scaler.transform(X_test[num_cols])

# ---------- MLflow Setup ----------
mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")
mlflow.set_experiment("Apartment_Rent_Prediction")

models = {
    "Linear Regression (Baseline)": {
        "model": LinearRegression(),
        "key": "linear",
        "params": {"fit_intercept": True}
    },
    "Decision Tree (Tuned)": {
        "model": DecisionTreeRegressor(max_depth=16, min_samples_leaf=10, random_state=42),
        "key": "decision_tree",
        "params": {"max_depth": 16, "min_samples_leaf": 10, "random_state": 42}
    },
    "Random Forest (100 Trees)": {
        "model": RandomForestRegressor(n_estimators=100, max_depth=16, min_samples_leaf=4, random_state=42, n_jobs=-1),
        "key": "random_forest",
        "params": {"n_estimators": 100, "max_depth": 16, "min_samples_leaf": 4, "random_state": 42}
    },
    "Hist Gradient Boosting (Tuned)": {
        "model": HistGradientBoostingRegressor(
            max_iter=500,
            learning_rate=0.08,
            max_leaf_nodes=127,
            min_samples_leaf=10,
            early_stopping=True,
            n_iter_no_change=15,
            validation_fraction=0.1,
            random_state=42
        ),
        "key": "hist_gradient_boosting",
        "params": {
            "max_iter": 500,
            "learning_rate": 0.08,
            "max_leaf_nodes": 127,
            "min_samples_leaf": 10,
            "early_stopping": True
        }
    }
}

metrics_summary = {}
mlflow_run_map = {}

print("Training and logging all 4 machine learning models to MLflow...")
for name, item in models.items():
    m = item["model"]
    k = item["key"]
    params = item.get("params", {})
    print(f"--> Training & Logging {name}...")
    
    with mlflow.start_run(run_name=f"Production_{name}") as run:
        # Fit model
        m.fit(X_train_scaled, y_train_log)
        pred_log = m.predict(X_test_scaled)
        pred = np.expm1(pred_log)
        
        # Calculate evaluation metrics
        mae_usd = mean_absolute_error(y_test_orig, pred)
        mse_usd = mean_squared_error(y_test_orig, pred)
        rmse_usd = np.sqrt(mse_usd)
        mape = mean_absolute_percentage_error(y_test_orig, pred) * 100
        r2 = r2_score(y_test_orig, pred)
        within_10 = (np.abs(y_test_orig.values - pred) / y_test_orig.values <= 0.10).mean() * 100
        within_20 = (np.abs(y_test_orig.values - pred) / y_test_orig.values <= 0.20).mean() * 100
        
        # Log parameters and metrics to MLflow
        mlflow.log_params(params)
        mlflow.log_metrics({
            "USD_MAE": float(mae_usd),
            "USD_RMSE": float(rmse_usd),
            "MAPE": float(mape),
            "R2": float(r2),
            "Within_10": float(within_10),
            "Within_20": float(within_20)
        })
        
        # Log trained model artifact to MLflow
        mlflow.sklearn.log_model(
            sk_model=m,
            name="model",
            registered_model_name=f"ApartmentRent_{k}"
        )
        
        metrics_summary[name] = {
            "key": k,
            "mae_usd": mae_usd,
            "mse_usd": mse_usd,
            "rmse_usd": rmse_usd,
            "mape": mape,
            "r2": r2,
            "within_10": within_10,
            "within_20": within_20,
            "mlflow_run_id": run.info.run_id
        }
        
        mlflow_run_map[name] = {
            "run_id": run.info.run_id,
            "model_key": k,
            "artifact_uri": f"runs:/{run.info.run_id}/model"
        }
        
        # Export compressed joblib artifact
        out_fpath = os.path.join(HERE, f"model_{k}.joblib")
        joblib.dump(m, out_fpath, compress=3)
        
        # Split large models (> 48MB) into safe < 50MB chunks for GitHub
        if os.path.getsize(out_fpath) > 48 * 1024 * 1024:
            chunk_size = 45 * 1024 * 1024
            with open(out_fpath, "rb") as f:
                part = 0
                while chunk := f.read(chunk_size):
                    with open(f"{out_fpath}.part{part}", "wb") as pf:
                        pf.write(chunk)
                    part += 1
            print(f"    [GitHub Safe Mode] Split {os.path.basename(out_fpath)} into {part} chunks (< 50MB each)")
            
        print(f"    {name} -> MAE: ${mae_usd:.2f} | RMSE: ${rmse_usd:.2f} | MAPE: {mape:.2f}% | R²: {r2:.4f} [MLflow Run: {run.info.run_id[:8]}]")

# Save default deployed model (HistGradientBoosting)
joblib.dump(models["Hist Gradient Boosting (Tuned)"]["model"], os.path.join(HERE, "rent_model.joblib"), compress=3)

# Save preprocessors and metadata locally
joblib.dump(scaler, os.path.join(HERE, "scaler.joblib"))
joblib.dump(num_cols, os.path.join(HERE, "num_cols.joblib"))
joblib.dump(list(X_train.columns), os.path.join(HERE, "model_columns.joblib"))
joblib.dump(metrics_summary, os.path.join(HERE, "model_metrics.joblib"))
joblib.dump(mlflow_run_map, os.path.join(HERE, "mlflow_run_map.joblib"))

geo = dc.groupby("state")[["latitude", "longitude"]].median()
joblib.dump(geo, os.path.join(HERE, "state_geo.joblib"))
joblib.dump({
    "city_medians": city_medians,
    "overall_median": overall_median,
    "city_table": city_grouped
}, os.path.join(HERE, "city_geo.joblib"))

# Log preprocessor artifacts to MLflow metadata run
with mlflow.start_run(run_name="Production_Metadata_Preprocessors") as meta_run:
    mlflow.log_artifact(os.path.join(HERE, "scaler.joblib"), artifact_path="preprocessors")
    mlflow.log_artifact(os.path.join(HERE, "num_cols.joblib"), artifact_path="metadata")
    mlflow.log_artifact(os.path.join(HERE, "model_columns.joblib"), artifact_path="metadata")
    mlflow.log_artifact(os.path.join(HERE, "model_metrics.joblib"), artifact_path="metrics")
    mlflow.log_artifact(os.path.join(HERE, "state_geo.joblib"), artifact_path="metadata")
    mlflow.log_artifact(os.path.join(HERE, "city_geo.joblib"), artifact_path="metadata")
    mlflow.log_artifact(os.path.join(HERE, "mlflow_run_map.joblib"), artifact_path="metadata")

print("\nSuccessfully trained, logged to MLflow Registry, and saved all artifacts!")
