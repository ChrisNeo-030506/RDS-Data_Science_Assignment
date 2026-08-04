"""
train_model.py
---------------
Trains the apartment-rent Gradient Boosting model and saves three artifacts
next to this script:
    rent_model.joblib      - the trained model
    model_columns.joblib   - training feature columns (to align one-hot at inference)
    state_geo.joblib       - per-state median latitude/longitude

Run ONCE inside the same environment that runs the Streamlit app:
    python train_model.py

Because the model is trained and saved with the LOCAL scikit-learn version,
this guarantees app.py can load it without any version-mismatch error.
"""
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

HERE = os.path.dirname(os.path.abspath(__file__))
CSV  = os.path.join(HERE, "..", "apartments_for_rent_classified_100K.csv")

# ---------- Load ----------
df = pd.read_csv(CSV, sep=";", encoding="cp1252", low_memory=False)

# ---------- Clean (identical to the notebook) ----------
dc = df[df["price_type"] == "Monthly"].copy()
dc = dc.drop(columns=["id", "category", "title", "body", "currency", "price_display",
                      "price_type", "address", "source", "time", "cityname"])

def iqr_bounds(s, k):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr

pl, ph = iqr_bounds(dc["price"], 3.0)
sl, sh = iqr_bounds(dc["square_feet"], 3.0)
dc = dc[(dc["price"] >= pl) & (dc["price"] <= ph) &
        (dc["square_feet"] >= sl) & (dc["square_feet"] <= sh)]

dc["amenities"]    = dc["amenities"].fillna("none")
dc["pets_allowed"] = dc["pets_allowed"].fillna("none")
dc["bathrooms"]    = dc["bathrooms"].fillna(dc["bathrooms"].median())
dc["bedrooms"]     = dc["bedrooms"].fillna(dc["bedrooms"].median())
dc = dc.dropna(subset=["state", "latitude", "longitude"])

dc["amenity_count"] = dc["amenities"].apply(lambda x: 0 if x == "none" else len(x.split(",")))
dc["pets_flag"]     = dc["pets_allowed"].apply(lambda x: 0 if x == "none" else 1)

# ---------- Encode ----------
dm = dc.drop(columns=["amenities", "pets_allowed"])
dm = pd.get_dummies(dm, columns=["fee", "has_photo", "state"], drop_first=True)

# ---------- Split ----------
X = dm.drop(columns=["price"])
y = dm["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------- Train (Hist Gradient Boosting: fast & compact on the 100K data) ----------
model = HistGradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print(f"Test  MAE {mean_absolute_error(y_test, pred):.2f} | "
      f"RMSE {np.sqrt(mean_squared_error(y_test, pred)):.2f} | "
      f"R2 {r2_score(y_test, pred):.4f}")

# ---------- Save artifacts next to this script ----------
joblib.dump(model, os.path.join(HERE, "rent_model.joblib"))
joblib.dump(list(X.columns), os.path.join(HERE, "model_columns.joblib"))
geo = dc.groupby("state")[["latitude", "longitude"]].median()
joblib.dump(geo, os.path.join(HERE, "state_geo.joblib"))
print("Saved: rent_model.joblib, model_columns.joblib, state_geo.joblib")
