import os
import glob
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
CSV  = os.path.join(ROOT, "apartments_for_rent_classified_100K.csv")

print("Loading dataset...")
df = pd.read_csv(CSV, sep=";", encoding="cp1252", low_memory=False)

# ---------- Step 3: Data Cleaning (Exact Notebook Alignment) ----------
# 1. Filter monthly rentals & drop unnecessary metadata
dc = df[df["price_type"] == "Monthly"].copy()
dc = dc.drop(columns=["id", "category", "title", "body", "currency", "price_display",
                      "price_type", "address", "source", "time", "pets_allowed"])

# 2. Drop rows missing essential features & median imputation
dc = dc.dropna(subset=["amenities", "cityname", "state", "latitude", "longitude"])
dc["bathrooms"] = dc["bathrooms"].fillna(dc["bathrooms"].median())
dc["bedrooms"]  = dc["bedrooms"].fillna(dc["bedrooms"].median())

# 3. IQR Outlier Detection (k=1.5 as in Notebook cell 49)
def iqr_bounds(series, k=1.5):
    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    IQR = Q3 - Q1
    return Q1 - k * IQR, Q3 + k * IQR

price_low, price_high = iqr_bounds(dc["price"], 1.5)
sqft_low, sqft_high   = iqr_bounds(dc["square_feet"], 1.5)

dc = dc[(dc["price"] >= price_low) & (dc["price"] <= price_high) &
        (dc["square_feet"] >= sqft_low) & (dc["square_feet"] <= sqft_high)]

# 4. Remove duplicates
dc = dc.drop_duplicates(keep="first")
print(f"Cleaned dataset shape: {dc.shape}")

# ---------- Step 4: Feature Engineering ----------
# 1. Binary amenity flags
amenities_lower = dc["amenities"].str.lower()
key_amenities = ["pool", "gym", "dishwasher", "parking", "garage", "washer",
                 "dryer", "ac", "patio", "gated", "fireplace", "elevator"]
for item in key_amenities:
    dc[f"has_{item}"] = amenities_lower.apply(lambda x: 1 if item in x else 0)

# 2. Amenity count (len(split(',')) as in Notebook cell 54)
dc["amenity_count"] = dc["amenities"].apply(lambda x: len(x.split(",")))

# 3. Domain-specific interaction ratios
dc["sqft_per_bedroom"]  = dc["square_feet"] / (dc["bedrooms"] + 1)
dc["sqft_per_bathroom"] = dc["square_feet"] / (dc["bathrooms"] + 1)
dc["bed_bath_ratio"]    = dc["bedrooms"] / (dc["bathrooms"] + 0.5)

# Save rich city metadata for UI map & selector
city_grouped = dc.groupby(["cityname", "state"]).agg({
    "price": "median",
    "latitude": "median",
    "longitude": "median"
}).reset_index()

# ---------- Train/Test Split (Zero Data Leakage) ----------
train_df, test_df = train_test_split(dc, test_size=0.2, random_state=42)

# Fit target encoding strictly on train partition
overall_median = train_df["price"].median()
city_medians = train_df.groupby("cityname")["price"].median().to_dict()

train_df["city_median_price"] = train_df["cityname"].map(city_medians).fillna(overall_median)
test_df["city_median_price"]  = test_df["cityname"].map(city_medians).fillna(overall_median)

# ---------- One-Hot Encoding ----------
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

models = {
    "Linear Regression (Baseline)": {
        "model": LinearRegression(),
        "key": "linear",
        "diag_key": "linear"
    },
    "Decision Tree (Tuned)": {
        "model": DecisionTreeRegressor(max_depth=16, min_samples_leaf=10, random_state=42),
        "key": "decision_tree",
        "diag_key": "dt"
    },
    "Random Forest (100 Trees)": {
        "model": RandomForestRegressor(n_estimators=100, max_depth=25, min_samples_leaf=1, random_state=42, n_jobs=-1),
        "key": "random_forest",
        "diag_key": "rf"
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
        "diag_key": "hgb"
    }
}

metrics_summary = {}

# Sample for diagnostic parquet
diag_sample = test_df.sample(min(5000, len(test_df)), random_state=42).copy()
diag_sample_scaled = X_test_scaled.loc[diag_sample.index]

print("\nTraining all 4 machine learning models...")
for name, item in models.items():
    m = item["model"]
    k = item["key"]
    dk = item["diag_key"]
    print(f"--> Training {name}...")
    
    # Fit model on log-transformed target
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
    
    metrics_summary[name] = {
        "key": k,
        "mae_usd": mae_usd,
        "mse_usd": mse_usd,
        "rmse_usd": rmse_usd,
        "mape": mape,
        "r2": r2,
        "within_10": within_10,
        "within_20": within_20
    }
    
    # Diagnostic predictions for UI Tab 3
    sample_pred_log = m.predict(diag_sample_scaled)
    sample_pred = np.expm1(sample_pred_log)
    diag_sample[f"pred_{dk}"] = sample_pred
    diag_sample[f"res_{dk}"] = diag_sample["price"] - sample_pred
    diag_sample[f"abs_err_{dk}"] = np.abs(diag_sample[f"res_{dk}"])
    diag_sample[f"pct_err_{dk}"] = (diag_sample[f"abs_err_{dk}"] / diag_sample["price"]) * 100
    
    # Export compressed joblib artifact
    out_fpath = os.path.join(HERE, f"model_{k}.joblib")
    if k == "random_forest":
        compact_forest = []
        for e in m.estimators_:
            t = e.tree_
            compact_forest.append({
                'left': t.children_left.astype(np.int32),
                'right': t.children_right.astype(np.int32),
                'feature': t.feature.astype(np.int16),
                'threshold': t.threshold.astype(np.float32),
                'value': t.value[:, 0, 0].astype(np.float32)
            })
        joblib.dump(compact_forest, out_fpath, compress=3)
    else:
        joblib.dump(m, out_fpath, compress=3)
    
    # Clean old split chunks if any exist
    for old_part in glob.glob(f"{out_fpath}.part*"):
        try:
            os.remove(old_part)
        except OSError:
            pass
            
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
        
    print(f"    {name} -> MAE: ${mae_usd:.2f} | RMSE: ${rmse_usd:.2f} | MAPE: {mape:.2f}% | R²: {r2:.4f} | Size: {os.path.getsize(out_fpath)/1024**2:.2f} MB")

# Save default deployed model (HistGradientBoosting)
hgb_model = models["Hist Gradient Boosting (Tuned)"]["model"]
joblib.dump(hgb_model, os.path.join(HERE, "rent_model.joblib"), compress=3)

# Save preprocessors and metadata locally
joblib.dump(scaler, os.path.join(HERE, "scaler.joblib"))
joblib.dump(num_cols, os.path.join(HERE, "num_cols.joblib"))
joblib.dump(list(X_train.columns), os.path.join(HERE, "model_columns.joblib"))
joblib.dump(metrics_summary, os.path.join(HERE, "model_metrics.joblib"))

geo = dc.groupby("state")[["latitude", "longitude"]].median()
joblib.dump(geo, os.path.join(HERE, "state_geo.joblib"))
joblib.dump({
    "city_medians": city_medians,
    "overall_median": overall_median,
    "city_table": city_grouped
}, os.path.join(HERE, "city_geo.joblib"))

# Sync root level artifacts
joblib.dump(hgb_model, os.path.join(ROOT, "rent_model.joblib"), compress=3)
joblib.dump(list(X_train.columns), os.path.join(ROOT, "model_columns.joblib"))
joblib.dump(geo, os.path.join(ROOT, "state_geo.joblib"))

# Save updated diagnostic parquet for Tab 3
diag_sample["price_per_sqft"] = diag_sample["price"] / diag_sample["square_feet"]
diag_sample["log_price"] = np.log1p(diag_sample["price"])
diag_parquet_path = os.path.join(HERE, "data", "model_diag_sample.parquet")
diag_sample.to_parquet(diag_parquet_path)
print(f"Saved diagnostic sample ({len(diag_sample)} rows) to {diag_parquet_path}")

print("\nSuccessfully trained and saved all production model artifacts!")
