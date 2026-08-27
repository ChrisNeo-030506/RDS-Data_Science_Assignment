import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------- Page Configuration ----------
st.set_page_config(
    page_title="US Apartment Rent Estimator",
    layout="wide",
    initial_sidebar_state="expanded"
)

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------- Custom Styling ----------
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        text-align: center;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 800;
        color: #0F172A;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    .hero-card {
        background: linear-gradient(135deg, #1E40AF 0%, #3B82F6 100%);
        color: white;
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 10px 25px -5px rgba(59, 130, 246, 0.4);
        margin-bottom: 24px;
    }
    .hero-price {
        font-size: 3rem;
        font-weight: 900;
        letter-spacing: -0.02em;
        margin: 10px 0;
    }
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-primary {
        background-color: rgba(255, 255, 255, 0.2);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

import mlflow
import mlflow.sklearn

DB_PATH = os.path.join(HERE, "..", "mlflow.db")
MLFLOW_URI = f"sqlite:///{DB_PATH}" if os.path.exists(DB_PATH) else f"sqlite:///{os.path.join(HERE, 'mlflow.db')}"

# ---------- Load Trained Artifacts (MLflow Registry + Local Fallbacks) ----------
@st.cache_resource
def load_all_artifacts():
    scaler = joblib.load(os.path.join(HERE, "scaler.joblib"))
    num_cols = joblib.load(os.path.join(HERE, "num_cols.joblib"))
    columns = joblib.load(os.path.join(HERE, "model_columns.joblib"))
    state_geo = joblib.load(os.path.join(HERE, "state_geo.joblib"))
    city_data = joblib.load(os.path.join(HERE, "city_geo.joblib"))
    
    models = {}
    mlflow_meta = {}
    
    # 1. Primary: Load models via MLflow Run Map & Registry
    try:
        run_map_path = os.path.join(HERE, "mlflow_run_map.joblib")
        if os.path.exists(run_map_path):
            mlflow.set_tracking_uri(MLFLOW_URI)
            run_map = joblib.load(run_map_path)
            
            disp_map = {
                "Hist Gradient Boosting (Tuned)": "Gradient Boosting (Recommended)",
                "Random Forest (100 Trees)": "Random Forest Ensemble",
                "Decision Tree (Tuned)": "Decision Tree",
                "Linear Regression (Baseline)": "Linear Baseline"
            }
            for train_name, meta in run_map.items():
                display_label = disp_map.get(train_name, train_name)
                try:
                    loaded_model = mlflow.sklearn.load_model(meta["artifact_uri"])
                    models[display_label] = loaded_model
                    mlflow_meta[display_label] = meta["run_id"]
                except Exception:
                    pass
    except Exception:
        pass
        
    # 2. Secondary: Fallback to local .joblib files
    candidate_joblibs = [
        ("Gradient Boosting (Recommended)", "model_hist_gradient_boosting.joblib"),
        ("Random Forest Ensemble", "model_random_forest.joblib"),
        ("Decision Tree", "model_decision_tree.joblib"),
        ("Linear Baseline", "model_linear.joblib")
    ]
    for display_name, file_name in candidate_joblibs:
        if display_name not in models:
            f_path = os.path.join(HERE, file_name)
            if os.path.exists(f_path):
                try:
                    models[display_name] = joblib.load(f_path)
                except Exception:
                    pass
                    
    # 3. Fallback to default rent_model.joblib if needed
    if not models:
        default_model_path = os.path.join(HERE, "rent_model.joblib")
        if os.path.exists(default_model_path):
            models["Gradient Boosting (Recommended)"] = joblib.load(default_model_path)
            
    return models, scaler, num_cols, columns, state_geo, city_data, mlflow_meta

models, scaler, num_cols, columns, state_geo, city_data, mlflow_meta = load_all_artifacts()

states = sorted(state_geo.index)
city_table = city_data.get("city_table", pd.DataFrame())

# ---------- Sidebar Inputs ----------
st.sidebar.markdown("## Property and Estimation Settings")

# Model Selection
model_options = list(models.keys()) if models else ["Gradient Boosting (Recommended)"]
selected_option = st.sidebar.selectbox("Valuation Algorithm", model_options, index=0)


st.sidebar.markdown("---")
st.sidebar.markdown("### Location")

# State Selection
selected_state = st.sidebar.selectbox("State", states, index=states.index("CA") if "CA" in states else 0)

# Dynamic City Selection based on State
if not city_table.empty and selected_state in city_table["state"].values:
    state_cities = sorted(city_table[city_table["state"] == selected_state]["cityname"].unique())
    selected_city = st.sidebar.selectbox("City", state_cities, index=0)
    city_row = city_table[(city_table["state"] == selected_state) & (city_table["cityname"] == selected_city)].iloc[0]
    lat_val = float(city_row["latitude"])
    lon_val = float(city_row["longitude"])
    city_med = float(city_row["price"])
else:
    selected_city = "State Average"
    lat_val = float(state_geo.loc[selected_state, "latitude"])
    lon_val = float(state_geo.loc[selected_state, "longitude"])
    city_med = float(city_data.get("overall_median", 1350.0))

st.sidebar.caption(f"Local Market Median: ${city_med:,.0f} / month")

st.sidebar.markdown("---")
st.sidebar.markdown("### Space and Layout")

square_feet = st.sidebar.slider("Living Area (Square Feet)", 150, 4500, 950, step=25)
bedrooms = st.sidebar.slider("Bedrooms", 0, 6, 2)
bathrooms = st.sidebar.slider("Bathrooms", 1.0, 5.0, 1.5, step=0.5, format="%g")

st.sidebar.markdown("---")
st.sidebar.markdown("### Listing Features")
has_photo = st.sidebar.selectbox("Photo Listing", ["Yes", "Thumbnail", "No"], index=0)
fee = st.sidebar.selectbox("Broker Fee Required", ["No", "Yes"], index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("### Amenities")

with st.sidebar.expander("Select Available Amenities", expanded=True):
    col_a, col_b = st.columns(2)
    with col_a:
        has_washer     = st.checkbox("In-Unit Washer", value=True)
        has_dryer      = st.checkbox("Dryer", value=True)
        has_ac         = st.checkbox("Air Conditioning", value=True)
        has_parking    = st.checkbox("Parking Space", value=True)
        has_garage     = st.checkbox("Garage", value=False)
        has_dishwasher = st.checkbox("Dishwasher", value=True)
    with col_b:
        has_pool       = st.checkbox("Swimming Pool", value=False)
        has_gym        = st.checkbox("Fitness Center", value=False)
        has_elevator   = st.checkbox("Elevator", value=False)
        has_patio      = st.checkbox("Balcony / Patio", value=False)
        has_gated      = st.checkbox("Gated Community", value=False)
        has_fireplace  = st.checkbox("Fireplace", value=False)

amenities_dict = {
    "has_pool": has_pool, "has_gym": has_gym, "has_dishwasher": has_dishwasher,
    "has_parking": has_parking, "has_garage": has_garage, "has_washer": has_washer,
    "has_dryer": has_dryer, "has_ac": has_ac, "has_patio": has_patio,
    "has_gated": has_gated, "has_fireplace": has_fireplace, "has_elevator": has_elevator
}
amenity_count = sum(amenities_dict.values())

# ---------- Feature Matrix Builder ----------
def build_feature_vector():
    row = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)
    
    # Core numericals
    row["square_feet"]   = square_feet
    row["bedrooms"]      = bedrooms
    row["bathrooms"]     = bathrooms
    row["amenity_count"] = amenity_count
    
    # Amenities
    for col, val in amenities_dict.items():
        if col in row.columns:
            row[col] = 1 if val else 0
            
    # Space & Interaction ratios
    row["sqft_per_bedroom"]  = square_feet / (bedrooms + 1)
    row["sqft_per_bathroom"] = square_feet / (bathrooms + 1)
    row["bed_bath_ratio"]    = bedrooms / (bathrooms + 0.5)
    
    # Geographical & Target Encoding
    row["latitude"]          = lat_val
    row["longitude"]         = lon_val
    row["city_median_price"] = city_med
    
    # One-hot encoded categories
    for col in (f"has_photo_{has_photo}", f"state_{selected_state}", f"fee_{fee}"):
        if col in row.columns:
            row[col] = 1

    # Standardize continuous columns
    row[num_cols] = scaler.transform(row[num_cols])
    return row

# ---------- Main Dashboard Header ----------
st.markdown("<div class='main-title'>US Apartment Rent Estimator</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Accurate, data-driven rental valuations powered by nationwide market analytics</div>", unsafe_allow_html=True)

# Overview Metric Strip
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"<div class='metric-card'><div class='metric-label'>Target Market</div><div class='metric-val'>{selected_city}, {selected_state}</div></div>", unsafe_allow_html=True)
with c2:
    st.markdown(f"<div class='metric-card'><div class='metric-label'>Living Area</div><div class='metric-val'>{square_feet:,} <span style='font-size:1rem;'>sq ft</span></div></div>", unsafe_allow_html=True)
with c3:
    st.markdown(f"<div class='metric-card'><div class='metric-label'>Layout</div><div class='metric-val'>{bedrooms} Bed · {bathrooms:g} Bath</div></div>", unsafe_allow_html=True)
with c4:
    st.markdown(f"<div class='metric-card'><div class='metric-label'>Amenities Included</div><div class='metric-val'>{amenity_count} <span style='font-size:1rem;'>/ 12</span></div></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------- Prediction Engine ----------
feat_row = build_feature_vector()

# Compute predictions across all models
all_preds = {}
for m_display, m_obj in models.items():
    pred_log = m_obj.predict(feat_row)[0]
    all_preds[m_display] = max(100.0, float(np.expm1(pred_log)))

# Determine Active Model
primary_model_name = selected_option


primary_pred = all_preds[primary_model_name]
lower_bound = primary_pred * 0.90
upper_bound = primary_pred * 1.10
price_per_sqft = primary_pred / square_feet

# Hero Prediction Card
st.markdown(f"""
<div class='hero-card'>
    <div class='badge badge-primary'>Selected Algorithm: {primary_model_name}</div>
    <div style='font-size: 1.1rem; opacity: 0.9; margin-top: 8px;'>Estimated Fair Market Monthly Rent</div>
    <div class='hero-price'>${primary_pred:,.0f} <span style='font-size:1.5rem; font-weight:400;'>/ month</span></div>
    <div style='font-size: 1.05rem; opacity: 0.95;'>
        Estimated Market Range: <b>${lower_bound:,.0f} – ${upper_bound:,.0f}</b> &nbsp;|&nbsp; 
        Rate: <b>${price_per_sqft:.2f} / sq ft</b>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------- Section 1: Multi-Model Comparison ----------
st.markdown("### Valuation Consensus Across Algorithms")

col_left, col_right = st.columns([3, 2])

with col_left:
    # Model Comparison Bar Chart
    comp_df = pd.DataFrame([
        {
            "Algorithm": k.split("(")[0].strip(),
            "Estimated Rent ($)": round(v)
        }
        for k, v in all_preds.items()
    ])
    
    st.bar_chart(data=comp_df.set_index("Algorithm")["Estimated Rent ($)"], color="#3B82F6", height=280)
    st.caption("Comparison showing rent predictions across different valuation models.")

with col_right:
    st.markdown("**Valuation Breakdown Table**")
    comp_table = []
    for k, v in all_preds.items():
        comp_table.append({
            "Algorithm": k,
            "Estimated Rent": f"${v:,.0f} / mo",
            "Expected Range": f"${v*0.90:,.0f} – ${v*1.10:,.0f}"
        })
    st.dataframe(pd.DataFrame(comp_table), use_container_width=True, hide_index=True)

# ---------- Section 2: Valuation Drivers & Metrics ----------
st.markdown("---")
st.markdown("### Property Valuation Summary")

c_drv1, c_drv2 = st.columns(2)
with c_drv1:
    st.write(f"- **Local Market Base**: Average rent for listings in **{selected_city}, {selected_state}** is **${city_med:,.0f}**.")
    st.write(f"- **Space Allocation**: **{square_feet / (bedrooms + 1):.0f} sq ft** per bedroom.")
    st.write(f"- **Broker Terms**: {'Broker fee required' if fee == 'Yes' else 'No broker fee (standard lease)'}.")
    st.write(f"- **Selected Features**: {amenity_count} out of 12 luxury amenities.")

with c_drv2:
    st.markdown("**Living Space Tier**")
    st.progress(min(1.0, square_feet / 3500))
    st.markdown("**Amenity Rating**")
    st.progress(amenity_count / 12)

# ---------- Footer ----------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #94A3B8; font-size: 0.85rem;'>"
    "US Apartment Rent Valuation Platform · Real Estate Market Analytics"
    "</div>",
    unsafe_allow_html=True
)
