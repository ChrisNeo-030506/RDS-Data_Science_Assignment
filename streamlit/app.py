import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------- Load trained artifacts (produced in the Colab notebook) ----------
@st.cache_resource
def load_artifacts():
    model   = joblib.load("rent_model.joblib")     # trained model (Hist Gradient Boosting)
    columns = joblib.load("model_columns.joblib")  # training feature columns (for alignment)
    geo     = joblib.load("state_geo.joblib")      # per-state median latitude/longitude
    return model, columns, geo

model, columns, geo = load_artifacts()
states = sorted(geo.index)

# ---------- Page ----------
st.title("🏠 Apartment Rent Predictor")
st.caption("Hist Gradient Boosting · trained on 100K listings (Test R² ≈ 0.77) — BMDS2003 Group 4")
st.write("Set the apartment features in the sidebar, then click **Predict rent**.")

# ---------- Sidebar inputs ----------
st.sidebar.header("Apartment features")
square_feet   = st.sidebar.slider("Square feet", 100, 5000, 900, step=10)
bedrooms      = st.sidebar.slider("Bedrooms", 0, 6, 2)
bathrooms     = st.sidebar.slider("Bathrooms", 1, 5, 1)
amenity_count = st.sidebar.slider("Number of amenities", 0, 15, 3)
pets_allowed  = st.sidebar.checkbox("Pets allowed", value=True)
has_photo     = st.sidebar.selectbox("Listing photo", ["No", "Thumbnail", "Yes"], index=2)
fee           = st.sidebar.selectbox("Broker fee", ["No", "Yes"], index=0)
state         = st.sidebar.selectbox("State", states, index=states.index("CA"))

# ---------- Build a single input row that matches the training columns ----------
def build_row():
    row = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)
    row["square_feet"]   = square_feet
    row["bedrooms"]      = bedrooms
    row["bathrooms"]     = bathrooms
    row["amenity_count"] = amenity_count
    row["pets_flag"]     = 1 if pets_allowed else 0
    # Location = the state's median lat/long (longitude & latitude are the model's top features)
    row["latitude"]  = geo.loc[state, "latitude"]
    row["longitude"] = geo.loc[state, "longitude"]
    # One-hot columns: set to 1 only if that column exists; baseline categories stay all-zero
    for col in (f"has_photo_{has_photo}", f"state_{state}", f"fee_{fee}"):
        if col in row.columns:
            row[col] = 1
    return row

# ---------- Predict ----------
if st.button("Predict rent", type="primary"):
    pred = model.predict(build_row())[0]
    st.success(f"Estimated monthly rent:  ${pred:,.0f}")
    st.caption(f"Location basis: {state}  (lat {geo.loc[state,'latitude']:.2f}, "
               f"lon {geo.loc[state,'longitude']:.2f})")
