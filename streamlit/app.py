import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import glob
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =====================================================================
#                        PAGE CONFIGURATION
# =====================================================================
st.set_page_config(
    page_title="US Apartment Rent Valuation & Intelligence Platform",
    page_icon=":material/apartment:",
    layout="wide",
    initial_sidebar_state="expanded"
)

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(HERE, "..")
DATA_DIR = os.path.join(HERE, "data")
EDA_DATA_PATH = os.path.join(DATA_DIR, "eda_clean_data.parquet")
DIAG_DATA_PATH = os.path.join(DATA_DIR, "model_diag_sample.parquet")

# =====================================================================
#                     PLOTLY THEME & COLOR SYSTEM
# =====================================================================
PLOTLY_TEMPLATE = "plotly_white"
COLOR_PRIMARY = "#2563EB"     # Royal Blue
COLOR_SECONDARY = "#3B82F6"   # Blue
COLOR_SUCCESS = "#10B981"     # Emerald Green
COLOR_WARNING = "#F59E0B"     # Amber
COLOR_DANGER = "#EF4444"      # Rose / Red
COLOR_PURPLE = "#8B5CF6"      # Violet
COLOR_DARK = "#0F172A"        # Slate Dark
COLOR_CARD_BG = "#FFFFFF"

MODEL_COLORS = {
    "Linear Regression": "#64748B",
    "Decision Tree": "#F59E0B",
    "Random Forest": "#3B82F6",
    "Hist Gradient Boosting (Tuned)": "#10B981",
    "Linear Baseline": "#64748B",
    "Decision Tree": "#F59E0B",
    "Random Forest Ensemble": "#3B82F6",
    "Gradient Boosting": "#10B981"
}

def apply_plotly_styling(fig, height=420, title=""):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        title=dict(
            text=f"<b>{title}</b>" if title else "",
            font=dict(size=14, color="#F8FAFC", family="Plus Jakarta Sans, Inter, sans-serif"),
            x=0,
            y=0.98,
            xanchor="left",
            yanchor="top"
        ),
        margin=dict(l=40, r=40, t=55 if title else 25, b=55),
        font=dict(family="Inter, sans-serif", color="#CBD5E1"),
        plot_bgcolor="rgba(15, 23, 42, 0.4)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.20,
            xanchor="center",
            x=0.5,
            font=dict(size=11, color="#CBD5E1")
        ),
        hoverlabel=dict(
            bgcolor="#1E293B",
            font_size=12,
            font_family="Inter, sans-serif",
            font_color="#FFFFFF"
        )
    )
    fig.update_traces(textangle=0, selector=dict(type="bar"))
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="rgba(148, 163, 184, 0.15)",
        tickfont=dict(color="#94A3B8"),
        title_font=dict(color="#E2E8F0")
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="rgba(148, 163, 184, 0.15)",
        tickfont=dict(color="#94A3B8"),
        title_font=dict(color="#E2E8F0")
    )
    return fig

# =====================================================================
#                     HIGH-CLASS CUSTOM CSS
# =====================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');
    @import url('https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3, h4 {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 700 !important;
    }
    h1 {
        color: #F8FAFC !important;
    }
    h2 {
        color: #60A5FA !important;
        font-size: 1.5rem !important;
    }
    h3 {
        color: #38BDF8 !important;
        font-size: 1.25rem !important;
        margin-top: 1.2rem !important;
    }
    h4 {
        color: #E2E8F0 !important;
        font-size: 1.08rem !important;
        margin-top: 1.0rem !important;
    }

    /* Main Page Heading with Vibrant Gradient */
    .main-title {
        font-size: 2.25rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin-bottom: 0.3rem;
        background: linear-gradient(135deg, #60A5FA 0%, #38BDF8 40%, #A78BFA 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: inline-block;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    .sub-title {
        font-size: 1.02rem;
        color: #94A3B8;
        margin-bottom: 1.5rem;
        font-weight: 400;
    }

    /* Section Headers with Glow */
    .section-header {
        font-size: 1.35rem;
        font-weight: 700;
        color: #38BDF8;
        margin: 1.6rem 0 0.8rem 0;
        display: flex;
        align-items: center;
        gap: 8px;
        letter-spacing: -0.01em;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    .eda-section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #38BDF8;
        padding-bottom: 8px;
        margin: 20px 0 16px 0;
        border-bottom: 1px solid rgba(56, 189, 248, 0.22);
        letter-spacing: -0.01em;
        font-family: 'Plus Jakarta Sans', sans-serif;       
    }

    /* Premium Glassmorphic Metric Cards with Uniform Fixed-Height & Flexbox Alignment */
    .metric-card {
        background: rgba(30, 41, 59, 0.65);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 14px;
        padding: 16px 14px;
        box-shadow: 0 4px 14px -2px rgba(0, 0, 0, 0.25);
        backdrop-filter: blur(12px);
        text-align: center;
        min-height: 98px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        box-sizing: border-box;
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(96, 165, 250, 0.5);
        box-shadow: 0 8px 20px -4px rgba(37, 99, 235, 0.2);
    }
    .metric-val {
        font-size: 1.35rem;
        font-weight: 700;
        color: #F8FAFC;
        font-family: 'Plus Jakarta Sans', sans-serif;
        letter-spacing: -0.02em;
        line-height: 1.25;
        max-width: 100%;
        overflow-wrap: break-word;
        word-break: break-word;
    }
    .metric-label {
        font-size: 0.74rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
        margin-bottom: 6px;
    }

    /* Hero Prediction Card */
    .hero-card {
        background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 55%, #3B82F6 100%);
        color: white;
        padding: 24px 28px;
        border-radius: 18px;
        border: 1px solid rgba(255, 255, 255, 0.15);
        box-shadow: 0 16px 24px -6px rgba(37, 99, 235, 0.35);
        margin: 18px 0 24px 0;
        position: relative;
        overflow: hidden;
    }
    .hero-price {
        font-size: 2.85rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin: 8px 0 10px 0;
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #FFFFFF;
    }
    .badge {
        display: inline-block;
        padding: 5px 14px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .badge-primary {
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        color: #FFFFFF;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    .badge-success {
        background: rgba(16, 185, 129, 0.2);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }


    /* Tabs Styling — Luxury Frosted Glass Segmented Pill Bar */
    [data-testid="stTabs"] {
        margin-bottom: 24px;
    }
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 8px !important;
        background: rgba(15, 23, 42, 0.65) !important;
        padding: 6px 8px !important;
        border-radius: 14px !important;
        border: 1px solid rgba(148, 163, 184, 0.15) !important;
        box-shadow: 0 4px 16px -2px rgba(0, 0, 0, 0.3) !important;
        backdrop-filter: blur(16px) !important;
        display: flex !important;
        align-items: center !important;
    }
    [data-testid="stTabs"] [data-baseweb="tab"] {
        height: 40px !important;
        border-radius: 10px !important;
        font-weight: 500 !important;
        font-size: 0.92rem !important;
        padding: 6px 18px !important;
        background: transparent !important;
        border: 1px solid transparent !important;
        outline: none !important;
        box-shadow: none !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    [data-testid="stTabs"] [data-baseweb="tab"] * {
        color: #94A3B8 !important;
        font-size: 0.92rem !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stTabs"] [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.06) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    [data-testid="stTabs"] [data-baseweb="tab"]:hover * {
        color: #F1F5F9 !important;
    }
    /* Active Selected Tab — Sleek Frosted Sapphire Glow */
    [data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {
        background: rgba(37, 99, 235, 0.22) !important;
        border: 1px solid rgba(56, 189, 248, 0.5) !important;
        box-shadow: 0 0 16px rgba(56, 189, 248, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
    }
    [data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] * {
        color: #38BDF8 !important;
        font-weight: 700 !important;
        text-shadow: 0 0 10px rgba(56, 189, 248, 0.35) !important;
    }
    /* Suppress default Streamlit red underline line */
    [data-testid="stTabs"] [data-baseweb="tab-highlight"],
    [data-testid="stTabs"] [data-baseweb="tab-border"],
    div[data-baseweb="tab-highlight"],
    div[data-baseweb="tab-border"] {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        height: 0px !important;
        width: 0px !important;
        background: transparent !important;
        border: none !important;
    }
    /* EDA Card Container */
    .eda-card {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 16px -2px rgba(0, 0, 0, 0.25);
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .eda-card:hover {
        border-color: rgba(56, 189, 248, 0.35);
    }
    .eda-card-header {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        gap: 6px;
        margin-bottom: 10px;
    }
    .eda-pill {
        font-size: 0.68rem;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 9999px;
        background: rgba(56, 189, 248, 0.14);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.32);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        white-space: nowrap;
        display: inline-block;
    }
    .eda-card-title {
        font-size: 1.04rem;
        font-weight: 700;
        color: #F8FAFC;
        font-family: 'Plus Jakarta Sans', sans-serif;
        line-height: 1.35;
    }
    .insight-box {
        background: rgba(15, 23, 42, 0.65);
        border-left: 3px solid #38BDF8;
        border-radius: 0 8px 8px 0;
        padding: 10px 14px;
        margin-top: 10px;
        font-size: 0.86rem;
        color: #CBD5E1;
        line-height: 1.45;
    }
    .insight-box b {
        color: #38BDF8;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================================
#                   DATA & ARTIFACT LOADERS
# =====================================================================
@st.cache_resource
def load_models_and_artifacts():
    # Auto-reconstruct split binaries if needed
    for model_fname in ["model_random_forest.joblib", "model_hist_gradient_boosting.joblib"]:
        target_path = os.path.join(HERE, model_fname)
        if not os.path.exists(target_path):
            parts = sorted(glob.glob(f"{target_path}.part*"))
            if parts:
                with open(target_path, "wb") as out_f:
                    for p in parts:
                        with open(p, "rb") as in_f:
                            out_f.write(in_f.read())

    scaler = joblib.load(os.path.join(HERE, "scaler.joblib"))
    num_cols = joblib.load(os.path.join(HERE, "num_cols.joblib"))
    columns = joblib.load(os.path.join(HERE, "model_columns.joblib"))
    state_geo = joblib.load(os.path.join(HERE, "state_geo.joblib"))
    city_data = joblib.load(os.path.join(HERE, "city_geo.joblib"))

    models = {}
    candidate_joblibs = [
        ("Gradient Boosting", "model_hist_gradient_boosting.joblib"),
        ("Random Forest Ensemble", "model_random_forest.joblib"),
        ("Decision Tree", "model_decision_tree.joblib"),
        ("Linear Baseline", "model_linear.joblib")
    ]
    for display_name, file_name in candidate_joblibs:
        f_path = os.path.join(HERE, file_name)
        if os.path.exists(f_path):
            try:
                models[display_name] = joblib.load(f_path)
            except Exception:
                pass

    if "Gradient Boosting" not in models:
        default_model_path = os.path.join(HERE, "rent_model.joblib")
        if os.path.exists(default_model_path):
            try:
                models["Gradient Boosting"] = joblib.load(default_model_path)
            except Exception:
                pass

    metrics_path = os.path.join(HERE, "model_metrics.joblib")
    saved_metrics = joblib.load(metrics_path) if os.path.exists(metrics_path) else {}

    return models, scaler, num_cols, columns, state_geo, city_data, saved_metrics

@st.cache_data
def load_eda_dataset():
    if os.path.exists(EDA_DATA_PATH):
        return pd.read_parquet(EDA_DATA_PATH)
    return pd.DataFrame()

@st.cache_data
def load_diagnostic_dataset():
    if os.path.exists(DIAG_DATA_PATH):
        return pd.read_parquet(DIAG_DATA_PATH)
    return pd.DataFrame()

models, scaler, num_cols, columns, state_geo, city_data, saved_metrics = load_models_and_artifacts()
df_eda = load_eda_dataset()
df_diag = load_diagnostic_dataset()

states = sorted(state_geo.index)
city_table = city_data.get("city_table", pd.DataFrame())

# =====================================================================
#                          SIDEBAR CONTROLS
# =====================================================================
st.sidebar.markdown('## <i class="bi bi-sliders2" style="color:#38BDF8; vertical-align:middle; margin-right:6px;"></i> Property & Valuation Controls', unsafe_allow_html=True)

model_options = list(models.keys()) if models else ["Gradient Boosting"]
selected_option = st.sidebar.selectbox("Valuation Algorithm", model_options, index=0)

st.sidebar.markdown("---")
st.sidebar.markdown('### <i class="bi bi-geo-alt" style="color:#38BDF8; vertical-align:middle; margin-right:6px;"></i> Location', unsafe_allow_html=True)

selected_state = st.sidebar.selectbox("State", states, index=states.index("CA") if "CA" in states else 0)

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

st.sidebar.markdown(f'<div style="color:#94A3B8; font-size:0.88rem; margin:6px 0 12px 0;"><i class="bi bi-geo" style="color:#38BDF8;"></i> Local Market Median: <b>${city_med:,.0f} / mo</b></div>', unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown('### <i class="bi bi-aspect-ratio" style="color:#38BDF8; vertical-align:middle; margin-right:6px;"></i> Space & Layout', unsafe_allow_html=True)
square_feet = st.sidebar.slider("Living Area (Sq Ft)", 150, 4500, 950, step=25)
bedrooms = st.sidebar.slider("Bedrooms", 0, 6, 2)
bathrooms = st.sidebar.slider("Bathrooms", 1.0, 5.0, 1.5, step=0.5, format="%g")

st.sidebar.markdown("---")
st.sidebar.markdown('### <i class="bi bi-card-checklist" style="color:#38BDF8; vertical-align:middle; margin-right:6px;"></i> Listing Details', unsafe_allow_html=True)
has_photo = st.sidebar.selectbox("Photo Listing", ["Yes", "Thumbnail", "No"], index=0)
fee = st.sidebar.selectbox("Broker Fee Required", ["No", "Yes"], index=0)

st.sidebar.markdown("---")
st.sidebar.markdown('### <i class="bi bi-stars" style="color:#F59E0B; vertical-align:middle; margin-right:6px;"></i> Luxury Amenities', unsafe_allow_html=True)
with st.sidebar.expander("Configure Property Amenities", expanded=True):
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

# Feature vector constructor
def build_feature_vector(sqft=square_feet, beds=bedrooms, baths=bathrooms, amen_count=amenity_count, amen_map=amenities_dict):
    row = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)
    row["square_feet"] = sqft
    row["bedrooms"] = beds
    row["bathrooms"] = baths
    row["amenity_count"] = amen_count

    for col, val in amen_map.items():
        if col in row.columns:
            row[col] = 1 if val else 0

    row["sqft_per_bedroom"] = sqft / (beds + 1)
    row["sqft_per_bathroom"] = sqft / (baths + 1)
    row["bed_bath_ratio"] = beds / (baths + 0.5)
    row["latitude"] = lat_val
    row["longitude"] = lon_val
    row["city_median_price"] = city_med

    for col in (f"has_photo_{has_photo}", f"state_{selected_state}", f"fee_{fee}"):
        if col in row.columns:
            row[col] = 1

    row[num_cols] = scaler.transform(row[num_cols])
    return row

# =====================================================================
#                          APPLICATION HEADER
# =====================================================================
st.markdown("<div class='main-title'>US Apartment Rent Valuation & Intelligence Platform</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Interactive Data Science Suite · CRISP-DM Machine Learning Framework · Nationwide Rental Valuation</div>", unsafe_allow_html=True)

tab_predict, tab_eda, tab_models, tab_eval = st.tabs([
    "Valuation Studio",
    "Exploratory Data Analysis",
    "Model Deep-Dive & Diagnostics",
    "Performance & Evaluation"
])

# =====================================================================
#  TAB 1 — VALUATION STUDIO
# =====================================================================
with tab_predict:
    # Top KPI Strip
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"<div class='metric-card'><div class='metric-label'>Target Market</div><div class='metric-val'>{selected_city}, {selected_state}</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='metric-card'><div class='metric-label'>Living Area</div><div class='metric-val'>{square_feet:,} <span style='font-size:0.95rem; font-weight:500; color:#64748B;'>sq ft</span></div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='metric-card'><div class='metric-label'>Layout</div><div class='metric-val'>{bedrooms} Bed · {bathrooms:g} Bath</div></div>", unsafe_allow_html=True)
    with c4:
        st.markdown(f"<div class='metric-card'><div class='metric-label'>Luxury Amenities</div><div class='metric-val'>{amenity_count} <span style='font-size:0.95rem; font-weight:500; color:#64748B;'>/ 12</span></div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    feat_row = build_feature_vector()

    all_preds = {}
    for m_display, m_obj in models.items():
        try:
            pred_log = m_obj.predict(feat_row)[0]
            all_preds[m_display] = max(100.0, float(np.expm1(pred_log)))
        except Exception:
            pass

    if not all_preds:
        all_preds["Market Average Baseline"] = float(city_med)

    primary_model_name = selected_option if selected_option in all_preds else list(all_preds.keys())[0]
    primary_pred = all_preds.get(primary_model_name, float(city_med))
    lower_bound = primary_pred * 0.90
    upper_bound = primary_pred * 1.10
    price_per_sqft = primary_pred / square_feet

    # Hero Banner
    st.markdown(f"""
    <div class='hero-card'>
        <div class='badge badge-primary'>Selected Valuation Model: {primary_model_name}</div>
        <div style='font-size: 1.15rem; opacity: 0.9; margin-top: 10px;'>Estimated Fair Market Monthly Rent</div>
        <div class='hero-price'>${primary_pred:,.0f} <span style='font-size:1.6rem; font-weight:400;'>/ month</span></div>
        <div style='font-size: 1.05rem; opacity: 0.95;'>
            Expected Market Range: <b>${lower_bound:,.0f} – ${upper_bound:,.0f}</b> &nbsp;·&nbsp; 
            Unit Rate: <b>${price_per_sqft:.2f} / sq ft</b> &nbsp;·&nbsp;
            Local Base: <b>${city_med:,.0f} / mo</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Interactive Section: Multi-Model Consensus & Price Sensitivity
    col_chart, col_side = st.columns([3, 2])

    with col_chart:
        st.markdown("<div class='section-header'><i class='bi bi-bar-chart-steps'></i> Multi-Model Valuation Consensus</div>", unsafe_allow_html=True)
        comp_df = pd.DataFrame([
            {"Model": k, "Estimated Rent ($)": round(v), "Lower ($)": round(v*0.9), "Upper ($)": round(v*1.1)}
            for k, v in all_preds.items()
        ])

        fig_consensus = go.Figure()
        for idx, row in comp_df.iterrows():
            m_name = row["Model"]
            m_color = MODEL_COLORS.get(m_name, COLOR_PRIMARY)
            fig_consensus.add_trace(go.Bar(
                name=m_name,
                y=[m_name.split("(")[0].strip()],
                x=[row["Estimated Rent ($)"]],
                orientation='h',
                marker=dict(color=m_color),
                error_x=dict(
                    type='data',
                    symmetric=False,
                    array=[row["Upper ($)"] - row["Estimated Rent ($)"]],
                    arrayminus=[row["Estimated Rent ($)"] - row["Lower ($)"]],
                    color="#64748B"
                ),
                hovertemplate=f"<b>{m_name}</b><br>Estimate: <b>$%{{x:,.0f}}</b> / mo<br>±10% Range: $%{{customdata[0]:,.0f}} – $%{{customdata[1]:,.0f}}<extra></extra>",
                customdata=[[row["Lower ($)"], row["Upper ($)"]]]
            ))

        fig_consensus.update_layout(
            showlegend=False,
            xaxis_title="Estimated Monthly Rent ($ USD)",
            yaxis=dict(autorange="reversed")
        )
        apply_plotly_styling(fig_consensus, height=290)
        st.plotly_chart(fig_consensus, use_container_width=True)

    with col_side:
        st.markdown("<div class='section-header'><i class='bi bi-table'></i> Consensus Breakdown</div>", unsafe_allow_html=True)
        breakdown_records = []
        for k, v in all_preds.items():
            diff = v - primary_pred
            diff_str = f"{diff:+.0f}" if abs(diff) > 1 else "Base"
            breakdown_records.append({
                "Algorithm": k.split("(")[0].strip(),
                "Estimate": f"${v:,.0f}",
                "Range (±10%)": f"${v*0.90:,.0f} – ${v*1.10:,.0f}",
                "Δ vs Selected": diff_str
            })
        st.dataframe(pd.DataFrame(breakdown_records), use_container_width=True, hide_index=True)

    st.markdown("---")

    # Interactive Price Sensitivity Curve
    st.markdown("<div class='section-header'><i class='bi bi-graph-up-arrow'></i> Dynamic Living Space Price Curve</div>", unsafe_allow_html=True)
    st.caption("Explore how estimated rent scales across different unit sizes while preserving your selected layout and amenities.")

    sqft_range = np.linspace(300, 3500, 35)
    curve_preds = []
    active_model = models.get(primary_model_name, list(models.values())[0] if models else None)

    if active_model:
        for s in sqft_range:
            row_s = build_feature_vector(sqft=s)
            p_log = active_model.predict(row_s)[0]
            curve_preds.append(max(100.0, float(np.expm1(p_log))))

        fig_curve = go.Figure()
        fig_curve.add_trace(go.Scatter(
            x=sqft_range,
            y=curve_preds,
            mode='lines',
            line=dict(color=COLOR_PRIMARY, width=3.5),
            name="Valuation Curve",
            hovertemplate="Size: <b>%{x:,.0f} sq ft</b><br>Estimated Rent: <b>$%{y:,.0f}</b> / mo<extra></extra>"
        ))
        fig_curve.add_trace(go.Scatter(
            x=[square_feet],
            y=[primary_pred],
            mode='markers',
            marker=dict(size=14, color=COLOR_DANGER, line=dict(color="#FFFFFF", width=3)),
            name="Current Property",
            hovertemplate=f"<b>Your Configuration</b><br>Size: <b>{square_feet:,} sq ft</b><br>Rent: <b>${primary_pred:,.0f}</b><extra></extra>"
        ))
        fig_curve.update_layout(
            xaxis_title="Living Area (Square Feet)",
            yaxis_title="Estimated Rent ($ USD)"
        )
        apply_plotly_styling(fig_curve, height=330)
        st.plotly_chart(fig_curve, use_container_width=True)


# =====================================================================
#  TAB 2 — EXPLORATORY DATA ANALYSIS (COMPREHENSIVE ANALYTICS COCKPIT)
# =====================================================================
with tab_eda:
    if df_eda.empty:
        st.error("EDA dataset not loaded. Please ensure `eda_clean_data.parquet` exists.")
    else:
        # Precompute common metrics & quantiles
        p95 = df_eda["price"].quantile(0.95)
        p99 = df_eda["price"].quantile(0.99)
        sq99 = df_eda["square_feet"].quantile(0.99)
        med_price = df_eda["price"].median()
        p25 = df_eda["price"].quantile(0.25)
        p75 = df_eda["price"].quantile(0.75)
        avg_sqft = df_eda["square_feet"].mean()
        avg_ppsqft = (df_eda["price"] / df_eda["square_feet"]).mean()

        # --- Top KPI Ribbon Strip ---
        st.markdown("<div style='margin-bottom: 14px;'>", unsafe_allow_html=True)
        ek1, ek2, ek3, ek4, ek5 = st.columns(5)
        with ek1:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Clean Listings</div><div class='metric-val'>{len(df_eda):,}</div></div>", unsafe_allow_html=True)
        with ek2:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>National Median</div><div class='metric-val'>${med_price:,.0f} <span style='font-size:0.85rem; color:#94A3B8;'>/ mo</span></div></div>", unsafe_allow_html=True)
        with ek3:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>IQR Spread (25–75%)</div><div class='metric-val'>${p25:,.0f} – ${p75:,.0f}</div></div>", unsafe_allow_html=True)
        with ek4:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Avg Living Area</div><div class='metric-val'>{avg_sqft:,.0f} <span style='font-size:0.85rem; color:#94A3B8;'>sq ft</span></div></div>", unsafe_allow_html=True)
        with ek5:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Avg Unit Rate</div><div class='metric-val'>${avg_ppsqft:.2f} <span style='font-size:0.85rem; color:#94A3B8;'>/ sq ft</span></div></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # --- Sub-Navigation Tabs for EDA ---
        eda_sub1, eda_sub2, eda_sub3, eda_sub4 = st.tabs([
            "1. Univariate Distributions (Graphs 1–5)",
            "2. Pricing Drivers & Bivariate Patterns (Graphs 6–13)",
            "3. Spatial Intelligence & Multivariate (Graphs 14–15)",
            "4. Interactive Market Slice & Live Explorer"
        ])

        # =================================================================
        # SUB-TAB 1: UNIVARIATE ANALYSIS (GRAPHS 1 - 5)
        # =================================================================
        with eda_sub1:
            st.markdown("<div class='section-header'><i class='bi bi-bar-chart'></i> Single-Variable Distribution Profiles & Statistical Baseline</div>", unsafe_allow_html=True)
            st.caption("Inspect probability density, skewness properties, outlier boundaries, and categorical supply compositions across US rental units.")

            # Row 1: Graph 1 & Graph 2
            u_col1, u_col2 = st.columns(2)
            with u_col1:
                st.markdown("""
                <div class='eda-card'>
                    <div class='eda-card-header'>
                        <span class='eda-pill'>Target Normalization</span>
                        <div class='eda-card-title'>Graph 1: Log-Transformed Monthly Rent Distribution</div>
                    </div>
                """, unsafe_allow_html=True)
                log_prices = np.log1p(df_eda["price"].dropna())
                fig1 = go.Figure()
                fig1.add_trace(go.Histogram(
                    x=log_prices,
                    nbinsx=55,
                    marker=dict(color="#0D9488", line=dict(color="rgba(255,255,255,0.2)", width=0.5)),
                    name="ln(1 + Price)",
                    hovertemplate="Log Rent: <b>%{x:.3f}</b><br>Listing Count: <b>%{y:,}</b><extra></extra>"
                ))
                mean_log = log_prices.mean()
                fig1.add_vline(
                    x=mean_log,
                    line_dash="dash",
                    line_color="#EF4444",
                    annotation_text=f"Mean ln(Price) = {mean_log:.2f}",
                    annotation_font=dict(color="#EF4444", size=11)
                )
                fig1.update_layout(xaxis_title="Log Rent: ln(1 + Monthly Price)", yaxis_title="Number of Listings", bargap=0.04)
                apply_plotly_styling(fig1, height=330)
                st.plotly_chart(fig1, use_container_width=True)
                st.markdown("""
                    <div class='insight-box'>
                        💡 <b>Data Science & Business Insight:</b> Raw rental prices exhibit heavy positive skewness (skew = +3.12). Applying the natural logarithmic transformation $\\ln(1 + y)$ successfully converts the distribution into an approximate Gaussian bell curve (mean = 7.20), stabilizing residual variance (homoscedasticity) and boosting regression predictive power by over 14% $R^2$.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with u_col2:
                st.markdown("""
                <div class='eda-card'>
                    <div class='eda-card-header'>
                        <span class='eda-pill'>Raw Skewness</span>
                        <div class='eda-card-title'>Graph 2: Monthly Rent Distribution (Capped at Percentile)</div>
                    </div>
                """, unsafe_allow_html=True)
                q_choice = st.selectbox("Select Display Percentile Cutoff:", ["95th Percentile ($2,700)", "90th Percentile ($2,250)", "99th Percentile ($3,950)"], index=0, key="uni_q_choice")
                q_val = p95 if "95th" in q_choice else (df_eda["price"].quantile(0.90) if "90th" in q_choice else p99)
                df_q = df_eda[df_eda["price"] <= q_val]

                fig2 = go.Figure()
                fig2.add_trace(go.Histogram(
                    x=df_q["price"],
                    nbinsx=50,
                    marker=dict(color="#059669", line=dict(color="rgba(255,255,255,0.2)", width=0.5)),
                    name="Monthly Rent ($)",
                    hovertemplate="Monthly Rent: <b>$%{x:,.0f}</b><br>Listings: <b>%{y:,}</b><extra></extra>"
                ))
                fig2.add_vline(x=med_price, line_dash="dash", line_color="#38BDF8", annotation_text=f"Median = ${med_price:,.0f}", annotation_font=dict(color="#38BDF8", size=11))
                fig2.update_layout(xaxis_title="Monthly Rent ($ USD)", yaxis_title="Number of Listings", bargap=0.04)
                apply_plotly_styling(fig2, height=280)
                st.plotly_chart(fig2, use_container_width=True)
                st.markdown(f"""
                    <div class='insight-box'>
                        💡 <b>Data Science & Business Insight:</b> 95% of nationwide listings fall under <b>${p95:,.0f}/month</b>, with the single highest concentration between $900 and $1,600. The long upper tail reflects luxury urban units that require robust non-linear handling.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Row 2: Graph 3 & Graph 4
            u_col3, u_col4 = st.columns(2)
            with u_col3:
                st.markdown("""
                <div class='eda-card'>
                    <div class='eda-card-header'>
                        <span class='eda-pill'>IQR Thresholding</span>
                        <div class='eda-card-title'>Graph 3: Price and Square Feet Outlier Profiling</div>
                    </div>
                """, unsafe_allow_html=True)
                fig3 = make_subplots(rows=1, cols=2, subplot_titles=["Rent Price ($) Distribution", "Square Feet Distribution"])
                fig3.add_trace(go.Box(
                    y=df_eda["price"].dropna(),
                    name="Price ($)",
                    marker_color="#38BDF8",
                    boxpoints="outliers",
                    jitter=0.2
                ), row=1, col=1)
                fig3.add_trace(go.Box(
                    y=df_eda["square_feet"].dropna(),
                    name="Square Feet",
                    marker_color="#34D399",
                    boxpoints="outliers",
                    jitter=0.2
                ), row=1, col=2)
                fig3.update_layout(showlegend=False)
                apply_plotly_styling(fig3, height=330)
                st.plotly_chart(fig3, use_container_width=True)
                st.markdown("""
                    <div class='insight-box'>
                        💡 <b>Data Science & Business Insight:</b> Extreme anomalies exist in raw listings (e.g. data entry typos with $100k+ rents or 50,000 sq ft). Applying $3.0\\times$ IQR boundaries removes severe noise while retaining legitimate luxury listings without distorting standard errors.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with u_col4:
                st.markdown("""
                <div class='eda-card'>
                    <div class='eda-card-header'>
                        <span class='eda-pill'>Supply Characteristics</span>
                        <div class='eda-card-title'>Graph 4: Categorical Distributions (Fee & Photos)</div>
                    </div>
                """, unsafe_allow_html=True)
                fee_counts = df_eda["fee"].value_counts().reset_index()
                photo_counts = df_eda["has_photo"].value_counts().reset_index()

                fig4 = make_subplots(rows=1, cols=2, subplot_titles=["Broker Fee Required", "Photo Listing Availability"])
                fig4.add_trace(go.Bar(
                    x=fee_counts["fee"],
                    y=fee_counts["count"],
                    marker=dict(color=["#3B82F6", "#F59E0B"]),
                    text=fee_counts["count"],
                    texttemplate="%{y:,}",
                    textposition="auto",
                    name="Fee",
                    hovertemplate="Fee Required (%{x}): <b>%{y:,}</b><extra></extra>"
                ), row=1, col=1)
                fig4.add_trace(go.Bar(
                    x=photo_counts["has_photo"],
                    y=photo_counts["count"],
                    marker=dict(color=["#10B981", "#6366F1", "#EC4899"]),
                    text=photo_counts["count"],
                    texttemplate="%{y:,}",
                    textposition="auto",
                    name="Photo",
                    hovertemplate="Photo Status (%{x}): <b>%{y:,}</b><extra></extra>"
                ), row=1, col=2)
                fig4.update_layout(showlegend=False, hovermode="x unified")
                apply_plotly_styling(fig4, height=330)
                st.plotly_chart(fig4, use_container_width=True)
                st.markdown("""
                    <div class='insight-box'>
                        💡 <b>Data Science & Business Insight:</b> Over <b>98.2%</b> of rental listings in the US dataset require zero broker fee, and <b>>90%</b> include high-resolution photo assets, reflecting digital real-estate platform standards.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Row 3: Graph 5
            st.markdown("""
            <div class='eda-card'>
                <div class='eda-card-header'>
                    <span class='eda-pill'>Square Footage Density</span>
                    <div class='eda-card-title'>Graph 5: Living Area Distribution Across US Apartments</div>
                </div>
            """, unsafe_allow_html=True)
            fig5 = go.Figure()
            df_sqft = df_eda[df_eda["square_feet"] <= 4000]
            fig5.add_trace(go.Histogram(
                x=df_sqft["square_feet"],
                nbinsx=40,
                marker=dict(color="#3B82F6", line=dict(color="rgba(255,255,255,0.2)", width=0.5)),
                name="Living Area",
                hovertemplate="Size: <b>%{x:,.0f} sq ft</b><br>Listings: <b>%{y:,}</b><extra></extra>"
            ))
            fig5.add_vline(x=avg_sqft, line_dash="dash", line_color="#F59E0B", annotation_text=f"Mean = {avg_sqft:,.0f} sq ft", annotation_font=dict(color="#F59E0B", size=11))
            fig5.update_layout(xaxis_title="Living Area (Square Feet)", yaxis_title="Number of Listings", bargap=0.04)
            apply_plotly_styling(fig5, height=320)
            st.plotly_chart(fig5, use_container_width=True)
            st.markdown(f"""
                <div class='insight-box'>
                    💡 <b>Data Science & Business Insight:</b> The distribution of apartment sizes centers around an average of <b>{avg_sqft:,.0f} sq ft</b> with two distinct multimodal peaks at <b>750 sq ft</b> (standard 1-bedroom unit) and <b>1,000 sq ft</b> (standard 2-bedroom unit).
                </div>
            </div>
            """, unsafe_allow_html=True)

        # =================================================================
        # SUB-TAB 2: BIVARIATE ANALYSIS (GRAPHS 6 - 13)
        # =================================================================
        with eda_sub2:
            st.markdown("<div class='section-header'><i class='bi bi-diagram-3'></i> Valuation Drivers, Correlation Dynamics & Feature Interactions</div>", unsafe_allow_html=True)
            st.caption("Examine how physical space, room layouts, state geographies, amenities, and temporal factors drive apartment rental pricing.")

            # Row 4: Graph 6 & Graph 7
            b_col1, b_col2 = st.columns(2)
            with b_col1:
                st.markdown("""
                <div class='eda-card'>
                    <div class='eda-card-header'>
                        <span class='eda-pill'>Pearson Correlation Matrix</span>
                        <div class='eda-card-title'>Graph 6: Correlation Heatmap (Continuous Features)</div>
                    </div>
                """, unsafe_allow_html=True)
                num_cols_eda = ["price", "square_feet", "bathrooms", "bedrooms", "latitude", "longitude"]
                corr_eda = df_eda[num_cols_eda].dropna().corr()
                fig6 = px.imshow(
                    corr_eda,
                    text_auto=".2f",
                    color_continuous_scale="RdBu_r",
                    zmin=-1,
                    zmax=1,
                    aspect="auto"
                )
                apply_plotly_styling(fig6, height=350)
                st.plotly_chart(fig6, use_container_width=True)
                st.markdown("""
                    <div class='insight-box'>
                        💡 <b>Data Science & Business Insight:</b> <b>Square Feet (r = 0.58)</b> and <b>Bathrooms (r = 0.50)</b> exhibit the highest positive linear correlation with price. Latitude and longitude show low global linear correlation because geographic rent drivers are highly non-linear and localized into metropolitan clusters.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with b_col2:
                st.markdown("""
                <div class='eda-card'>
                    <div class='eda-card-header'>
                        <span class='eda-pill'>Bivariate Elasticity</span>
                        <div class='eda-card-title'>Graph 7: Rent Price vs. Living Area by Bedroom Count</div>
                    </div>
                """, unsafe_allow_html=True)
                sample_biv = df_eda[(df_eda["price"] < p99) & (df_eda["square_feet"] < sq99) & (df_eda["bedrooms"].isin([1, 2, 3]))].sample(min(3500, len(df_eda)), random_state=42)
                fig7 = px.scatter(
                    sample_biv,
                    x="square_feet",
                    y="price",
                    color="bedrooms",
                    color_continuous_scale="Viridis",
                    opacity=0.6,
                    hover_name="cityname",
                    hover_data={"state": True, "price": ":$.0f", "square_feet": True, "bathrooms": True}
                )
                fig7.update_layout(xaxis_title="Square Feet", yaxis_title="Monthly Rent ($ USD)", coloraxis_colorbar=dict(title="Bedrooms"))
                apply_plotly_styling(fig7, height=350)
                st.plotly_chart(fig7, use_container_width=True)
                st.markdown("""
                    <div class='insight-box'>
                        💡 <b>Data Science & Business Insight:</b> Clear monotonic upward slope confirms that monthly rent scales proportionally with living area across all bedroom tiers. At equivalent square footage, units with fewer bedrooms often command higher rent per room due to luxury open floorplans.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Row 5: Graph 8 & Graph 9
            b_col3, b_col4 = st.columns(2)
            with b_col3:
                st.markdown("""
                <div class='eda-card'>
                    <div class='eda-card-header'>
                        <span class='eda-pill'>2D Valuation Matrix</span>
                        <div class='eda-card-title'>Graph 8: Average Rent Heatmap by Bed & Bath Grid ($)</div>
                    </div>
                """, unsafe_allow_html=True)
                df_filtered_layout = df_eda[
                    (df_eda["bedrooms"].isin([0, 1, 2, 3, 4])) &
                    (df_eda["bathrooms"].isin([1, 1.5, 2, 2.5, 3]))
                ]
                price_matrix = df_filtered_layout.pivot_table(values="price", index="bedrooms", columns="bathrooms", aggfunc="mean")
                fig8 = px.imshow(
                    price_matrix,
                    text_auto=".0f",
                    color_continuous_scale="YlGnBu",
                    labels=dict(x="Number of Bathrooms", y="Number of Bedrooms", color="Average Rent ($)"),
                    aspect="auto"
                )
                fig8.update_layout(xaxis_title="Number of Bathrooms", yaxis_title="Number of Bedrooms")
                apply_plotly_styling(fig8, height=350)
                st.plotly_chart(fig8, use_container_width=True)
                st.markdown("""
                    <div class='insight-box'>
                        💡 <b>Data Science & Business Insight:</b> Rents scale steadily along both room axes: adding a second bathroom to a 2-bedroom unit increases average rent by approximately <b>+$260/month</b>, reflecting strong tenant demand for private ensuite layouts.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with b_col4:
                st.markdown("""
                <div class='eda-card'>
                    <div class='eda-card-header'>
                        <span class='eda-pill'>Supply Concentration</span>
                        <div class='eda-card-title'>Graph 9: Top 10 Most Common Apartment Layouts</div>
                    </div>
                """, unsafe_allow_html=True)
                df_eda_copy = df_eda.copy()
                df_eda_copy["beds_str"] = df_eda_copy["bedrooms"].dropna().astype(str).str.split(".").str[0]
                df_eda_copy["baths_str"] = df_eda_copy["bathrooms"].dropna().astype(str).str.replace(r"\.0$", "", regex=True)
                df_eda_copy["layout"] = df_eda_copy["beds_str"] + " Bed / " + df_eda_copy["baths_str"] + " Bath"
                top_layouts = df_eda_copy["layout"].value_counts().nlargest(10).reset_index()
                top_layouts.columns = ["Layout", "Count"]

                fig9 = px.bar(
                    top_layouts.sort_values("Count", ascending=True),
                    x="Count",
                    y="Layout",
                    orientation='h',
                    color="Count",
                    color_continuous_scale="Blues",
                    text="Count"
                )
                fig9.update_traces(
                    texttemplate="%{x:,}",
                    textposition="outside",
                    textangle=0,
                    cliponaxis=False,
                    textfont=dict(color="#F8FAFC", size=11)
                )
                fig9.update_layout(
                    xaxis_title="Number of Listings",
                    yaxis_title="Layout (Bed / Bath)",
                    xaxis=dict(range=[0, top_layouts["Count"].max() * 1.18])
                )
                apply_plotly_styling(fig9, height=350)
                st.plotly_chart(fig9, use_container_width=True)
                st.markdown("""
                    <div class='insight-box'>
                        💡 <b>Data Science & Business Insight:</b> <b>2 Bed / 2 Bath</b> and <b>1 Bed / 1 Bath</b> dominate the US rental market, comprising over <b>65%</b> of overall inventory. Developers prioritize these units due to maximum liquidity and tenant appeal.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Row 6: Graph 10 & Graph 11
            b_col5, b_col6 = st.columns(2)
            with b_col5:
                st.markdown("""
                <div class='eda-card'>
                    <div class='eda-card-header'>
                        <span class='eda-pill'>Interquartile Variance</span>
                        <div class='eda-card-title'>Graph 10: Rent Price Distribution by Bedroom Count</div>
                    </div>
                """, unsafe_allow_html=True)
                df_beds = df_eda[df_eda["bedrooms"].isin([0, 1, 2, 3, 4])]
                fig10 = px.box(
                    df_beds,
                    x="bedrooms",
                    y="price",
                    color="bedrooms",
                    points=False
                )
                fig10.update_layout(showlegend=False, xaxis_title="Number of Bedrooms", yaxis_title="Price (USD)")
                apply_plotly_styling(fig10, height=340)
                st.plotly_chart(fig10, use_container_width=True)
                st.markdown("""
                    <div class='insight-box'>
                        💡 <b>Data Science & Business Insight:</b> Both the median price and interquartile spread (IQR) increase consistently with bedroom count. Variance is highest in 3-bedroom and 4-bedroom homes due to greater diversity in luxury amenities and square footage.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with b_col6:
                st.markdown("""
                <div class='eda-card'>
                    <div class='eda-card-header'>
                        <span class='eda-pill'>Cross-Tabulation Matrix</span>
                        <div class='eda-card-title'>Graph 11: Photo Availability vs. Broker Fee (%)</div>
                    </div>
                """, unsafe_allow_html=True)
                ct = pd.crosstab(df_eda["has_photo"], df_eda["fee"], normalize="index") * 100
                fig11 = px.imshow(
                    ct,
                    text_auto=".1f",
                    color_continuous_scale="Blues",
                    labels=dict(x="Broker Fee Required", y="Photo Status", color="Percentage (%)"),
                    aspect="auto"
                )
                fig11.update_layout(xaxis_title="Broker Fee Required", yaxis_title="Photo Availability")
                apply_plotly_styling(fig11, height=340)
                st.plotly_chart(fig11, use_container_width=True)
                st.markdown("""
                    <div class='insight-box'>
                        💡 <b>Data Science & Business Insight:</b> Across all photo categories, the vast majority (>98%) of listings do not require broker fees, establishing that broker fees are rare exceptions in nationwide US rental platforms.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Row 7: Graph 12 & Graph 13
            b_col7, b_col8 = st.columns(2)
            with b_col7:
                st.markdown("""
                <div class='eda-card'>
                    <div class='eda-card-header'>
                        <span class='eda-pill'>Geographic State Pricing</span>
                        <div class='eda-card-title'>Graph 12: Average Monthly Rent by State (Top 15 Most Listed)</div>
                    </div>
                """, unsafe_allow_html=True)
                top_states = df_eda["state"].value_counts().nlargest(15).index
                df_top_states = df_eda[df_eda["state"].isin(top_states)]
                state_order = df_top_states.groupby("state")["price"].mean().sort_values(ascending=False).reset_index()

                fig12 = px.bar(
                    state_order,
                    x="state",
                    y="price",
                    color="price",
                    color_continuous_scale="Viridis",
                    text="price"
                )
                fig12.update_traces(texttemplate="$%{y:,.0f}", textposition="auto")
                fig12.update_layout(xaxis_title="State Code", yaxis_title="Average Monthly Rent ($ USD)")
                apply_plotly_styling(fig12, height=350)
                st.plotly_chart(fig12, use_container_width=True)
                st.markdown("""
                    <div class='insight-box'>
                        💡 <b>Data Science & Business Insight:</b> Coastal powerhouse states (<b>California ($2,140)</b>, <b>Massachusetts ($2,280)</b>, <b>New York ($2,090)</b>, <b>Washington ($1,850)</b>) command significant rent premiums over Midwest and Southern states, confirming the need for target-encoded city medians.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with b_col8:
                st.markdown("""
                <div class='eda-card'>
                    <div class='eda-card-header'>
                        <span class='eda-pill'>Temporal Seasonality</span>
                        <div class='eda-card-title'>Graph 13: Listing Publish Velocity by Day of Week & Source</div>
                    </div>
                """, unsafe_allow_html=True)
                df_time = df_eda.copy()
                df_time["datetime"] = pd.to_datetime(df_time["time"], unit="s", errors="coerce")
                df_time["day_of_week"] = df_time["datetime"].dt.day_name()
                weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                top_sources = df_time["source"].value_counts().nlargest(3).index
                df_time_plot = df_time[df_time["source"].isin(top_sources) & df_time["day_of_week"].notna()]
                time_counts = df_time_plot.groupby(["day_of_week", "source"]).size().reset_index(name="count")

                fig13 = px.bar(
                    time_counts,
                    x="day_of_week",
                    y="count",
                    color="source",
                    barmode="group",
                    category_orders={"day_of_week": weekday_order}
                )
                fig13.update_traces(hovertemplate="%{y:,} listings")
                fig13.update_layout(xaxis_title="Day of Week", yaxis_title="Number of Published Listings", hovermode="x unified")
                apply_plotly_styling(fig13, height=350)
                st.plotly_chart(fig13, use_container_width=True)
                st.markdown("""
                    <div class='insight-box'>
                        💡 <b>Data Science & Business Insight:</b> Midweek days (<b>Tuesday through Thursday</b>) exhibit the highest listing publish velocity, whereas weekends experience steep declines as real-estate property managers focus on open house tours.
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # =================================================================
        # SUB-TAB 3: MULTIVARIATE & GEOSPATIAL (GRAPHS 14 - 15)
        # =================================================================
        with eda_sub3:
            st.markdown("<div class='section-header'><i class='bi bi-globe-americas'></i> Geospatial Clustering & High-Dimensional Feature Interactions</div>", unsafe_allow_html=True)
            st.caption("Analyze continental coordinate pricing density and pairwise non-linear interactions across size, unit rate, and room layouts.")

            # Row 8: Graph 14 & Graph 15
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.markdown("""
                <div class='eda-card'>
                    <div class='eda-card-header'>
                        <span class='eda-pill'>Geospatial Density Map</span>
                        <div class='eda-card-title'>Graph 14: Continental US Geographical Pricing Clusters</div>
                    </div>
                """, unsafe_allow_html=True)
                geo_sample_size = st.slider("Map Sample Density:", 2000, 8000, 5000, step=1000, key="geo_sample_slider")
                spatial_df = df_eda[
                    (df_eda["latitude"] > 24) & (df_eda["latitude"] < 50) &
                    (df_eda["longitude"] > -125) & (df_eda["longitude"] < -65) &
                    (df_eda["price"] < p99)
                ].sample(min(geo_sample_size, len(df_eda)), random_state=42)

                fig14 = px.scatter(
                    spatial_df,
                    x="longitude",
                    y="latitude",
                    color="price",
                    color_continuous_scale="Viridis",
                    opacity=0.6,
                    hover_name="cityname",
                    hover_data={"state": True, "price": ":$.0f", "square_feet": True, "bedrooms": True}
                )
                fig14.update_layout(xaxis_title="Longitude (West ← → East)", yaxis_title="Latitude (South ← → North)", coloraxis_colorbar=dict(title="Rent ($ USD)"))
                apply_plotly_styling(fig14, height=390)
                st.plotly_chart(fig14, use_container_width=True)
                st.markdown("""
                    <div class='insight-box'>
                        💡 <b>Data Science & Business Insight:</b> Nationwide spatial plotting uncovers stark coastal price density clusters in California, the Northeast Corridor (NYC/Boston), and the Pacific Northwest, contrasting with affordable interior Midwest and Sunbelt metropolitan regions.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with m_col2:
                st.markdown("""
                <div class='eda-card'>
                    <div class='eda-card-header'>
                        <span class='eda-pill'>3×3 Multivariate Matrix</span>
                        <div class='eda-card-title'>Graph 15: Pairwise Continuous Interactions by Bedroom Layout</div>
                    </div>
                """, unsafe_allow_html=True)
                pair_sample = df_eda[
                    (df_eda["price"] < p99) &
                    (df_eda["square_feet"] < sq99) &
                    (df_eda["bedrooms"].isin([1, 2, 3]))
                ].sample(min(1400, len(df_eda)), random_state=42).copy()

                pair_sample["Bedroom Type"] = pair_sample["bedrooms"].astype(int).astype(str) + " Bed"
                pair_sample["Price/SqFt ($)"] = pair_sample["price"] / pair_sample["square_feet"]
                pair_sample["Monthly Rent ($)"] = pair_sample["price"]
                pair_sample["Square Feet"] = pair_sample["square_feet"]

                pair_colors = {"1 Bed": "#38BDF8", "2 Bed": "#34D399", "3 Bed": "#FB923C"}
                btypes = ["1 Bed", "2 Bed", "3 Bed"]
                vars_list = ["Monthly Rent ($)", "Square Feet", "Price/SqFt ($)"]

                fig15 = make_subplots(
                    rows=3, cols=3,
                    shared_xaxes=False,
                    shared_yaxes=False,
                    horizontal_spacing=0.07,
                    vertical_spacing=0.07
                )

                for i in range(3):
                    for j in range(3):
                        if j > i:
                            continue
                        var_x = vars_list[j]
                        var_y = vars_list[i]

                        if i == j:
                            for btype in btypes:
                                sub = pair_sample[pair_sample["Bedroom Type"] == btype]
                                fig15.add_trace(go.Histogram(
                                    x=sub[var_x],
                                    name=btype,
                                    marker=dict(color=pair_colors[btype]),
                                    opacity=0.55,
                                    showlegend=(i == 0 and j == 0),
                                    legendgroup=btype,
                                    hovertemplate=f"<b>{btype}</b><br>{var_x}: %{{x:.1f}}<br>Count: %{{y}}<extra></extra>"
                                ), row=i+1, col=j+1)
                        else:
                            for btype in btypes:
                                sub = pair_sample[pair_sample["Bedroom Type"] == btype]
                                fig15.add_trace(go.Scatter(
                                    x=sub[var_x],
                                    y=sub[var_y],
                                    mode="markers",
                                    name=btype,
                                    marker=dict(color=pair_colors[btype], size=4, opacity=0.4),
                                    showlegend=False,
                                    legendgroup=btype,
                                    hovertemplate=f"<b>{btype}</b><br>{var_x}: %{{x:.1f}}<br>{var_y}: %{{y:.1f}}<extra></extra>"
                                ), row=i+1, col=j+1)

                        if i == 2:
                            fig15.update_xaxes(title_text=var_x, title_font=dict(size=9), row=i+1, col=j+1)
                        if j == 0:
                            fig15.update_yaxes(title_text=var_y if i != 0 else "Density", title_font=dict(size=9), row=i+1, col=j+1)

                fig15.update_layout(
                    barmode="overlay",
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(15,23,42,0.6)",
                    height=390,
                    margin=dict(l=35, r=10, t=25, b=35),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10))
                )
                st.plotly_chart(fig15, use_container_width=True)
                st.markdown("""
                    <div class='insight-box'>
                        💡 <b>Data Science & Business Insight:</b> The 3×3 pairwise matrix verifies distinct cluster boundaries: 1-bedroom apartments have higher unit density ($/sq ft) than 3-bedroom units, validating our engineered interaction term `sqft_per_bedroom`.
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # =================================================================
        # SUB-TAB 4: INTERACTIVE MARKET SLICE & LIVE DRILLDOWN
        # =================================================================
        with eda_sub4:
            st.markdown("<div class='section-header'><i class='bi bi-funnel'></i> Dynamic Market Slicing & Custom Drilldown Engine</div>", unsafe_allow_html=True)
            st.caption("Slice, filter, and dissect real-time rental micro-markets with customized multi-attribute parameters.")

            # Filter Controls
            f_col1, f_col2, f_col3, f_col4 = st.columns(4)
            with f_col1:
                available_states = sorted(df_eda["state"].dropna().unique().tolist())
                default_states = [s for s in ["CA", "TX", "FL", "NY", "WA"] if s in available_states]
                selected_filter_states = st.multiselect("Filter States:", available_states, default=default_states, key="slice_states")
            with f_col2:
                selected_price_range = st.slider("Monthly Rent Range ($):", 300, 5000, (600, 3000), step=50, key="slice_price")
            with f_col3:
                selected_beds = st.multiselect("Bedrooms:", [0, 1, 2, 3, 4], default=[1, 2, 3], key="slice_beds")
            with f_col4:
                require_laundry = st.checkbox("Require In-Unit Washer/Dryer", value=False, key="slice_laundry")
                require_parking = st.checkbox("Require Parking Space", value=False, key="slice_parking")

            # Apply Filter Mask
            mask = (
                (df_eda["price"].between(selected_price_range[0], selected_price_range[1])) &
                (df_eda["bedrooms"].isin(selected_beds) if selected_beds else True)
            )
            if selected_filter_states:
                mask = mask & (df_eda["state"].isin(selected_filter_states))
            if require_laundry:
                mask = mask & ((df_eda["has_washer"] == 1) | (df_eda["has_dryer"] == 1))
            if require_parking:
                mask = mask & ((df_eda["has_parking"] == 1) | (df_eda["has_garage"] == 1))

            df_slice = df_eda[mask]

            # Dynamic Metrics Strip for Filtered Slice
            st.markdown("<br>", unsafe_allow_html=True)
            s_m1, s_m2, s_m3, s_m4 = st.columns(4)
            slice_count = len(df_slice)
            slice_med = df_slice["price"].median() if slice_count > 0 else 0
            slice_sqft = df_slice["square_feet"].mean() if slice_count > 0 else 0
            slice_rate = (df_slice["price"] / df_slice["square_feet"]).mean() if slice_count > 0 else 0
            diff_med = slice_med - med_price

            s_m1.markdown(f"<div class='metric-card'><div class='metric-label'>Matching Inventory</div><div class='metric-val'>{slice_count:,} <span style='font-size:0.82rem; color:#94A3B8;'>({slice_count/len(df_eda)*100:.1f}%)</span></div></div>", unsafe_allow_html=True)
            s_m2.markdown(f"<div class='metric-card'><div class='metric-label'>Slice Median Rent</div><div class='metric-val'>${slice_med:,.0f} <span style='font-size:0.82rem; color:{'#10B981' if diff_med>=0 else '#EF4444'};'>({diff_med:+.0f} vs US)</span></div></div>", unsafe_allow_html=True)
            s_m3.markdown(f"<div class='metric-card'><div class='metric-label'>Avg Slice Living Area</div><div class='metric-val'>{slice_sqft:,.0f} <span style='font-size:0.82rem; color:#94A3B8;'>sq ft</span></div></div>", unsafe_allow_html=True)
            s_m4.markdown(f"<div class='metric-card'><div class='metric-label'>Avg Price per Sq Ft</div><div class='metric-val'>${slice_rate:.2f} <span style='font-size:0.82rem; color:#94A3B8;'>/ sq ft</span></div></div>", unsafe_allow_html=True)

            if slice_count > 0:
                # Live Interactive Charts for Filtered Slice
                sc_col1, sc_col2 = st.columns(2)
                with sc_col1:
                    st.markdown("#### Live Price vs. Size Scatter (Selected Slice)")
                    sample_slice = df_slice.sample(min(2000, len(df_slice)), random_state=42)
                    fig_slice_scatter = px.scatter(
                        sample_slice,
                        x="square_feet",
                        y="price",
                        color="bedrooms",
                        color_continuous_scale="Viridis",
                        hover_name="cityname",
                        hover_data={"state": True, "price": ":$.0f", "square_feet": True}
                    )
                    fig_slice_scatter.update_layout(xaxis_title="Square Feet", yaxis_title="Monthly Rent ($ USD)", coloraxis_colorbar=dict(title="Bedrooms"))
                    apply_plotly_styling(fig_slice_scatter, height=350)
                    st.plotly_chart(fig_slice_scatter, use_container_width=True)

                with sc_col2:
                    st.markdown("#### Top 10 Cities by Volume in Filtered Slice")
                    top_cities_slice = df_slice["cityname"].value_counts().nlargest(10).reset_index()
                    top_cities_slice.columns = ["City", "Listings"]
                    city_avg = df_slice[df_slice["cityname"].isin(top_cities_slice["City"])].groupby("cityname")["price"].mean().reset_index()
                    city_avg.columns = ["City", "AvgPrice"]
                    top_cities_slice = top_cities_slice.merge(city_avg, on="City")

                    fig_city_slice = px.bar(
                        top_cities_slice,
                        x="City",
                        y="Listings",
                        color="AvgPrice",
                        color_continuous_scale="Blues",
                        text="Listings",
                        hover_data={"AvgPrice": ":$.0f"}
                    )
                    fig_city_slice.update_traces(textposition="auto")
                    fig_city_slice.update_layout(xaxis_title="City", yaxis_title="Listings in Slice", coloraxis_colorbar=dict(title="Avg Rent ($)"))
                    apply_plotly_styling(fig_city_slice, height=350)
                    st.plotly_chart(fig_city_slice, use_container_width=True)
            else:
                st.warning("No listings match your selected filter criteria. Please broaden your state, price, or room filters.")


# =====================================================================
#  TAB 3 — MODEL ANALYSIS & DIAGNOSTICS
# =====================================================================
with tab_models:
    selected_diag_model = st.selectbox(
        "Select Model Architecture for Diagnostic Inspection:",
        [
            "Model 1: Linear Regression",
            "Model 2: Decision Tree",
            "Model 3: Random Forest",
            "Model 4: Hist Gradient Boosting"
        ],
        index=3
    )

    model_key_map = {
        "Model 1: Linear Regression": ("linear", "Linear Regression", "#64748B"),
        "Model 2: Decision Tree": ("dt", "Decision Tree", "#F59E0B"),
        "Model 3: Random Forest": ("rf", "Random Forest", "#3B82F6"),
        "Model 4: Hist Gradient Boosting": ("hgb", "Hist Gradient Boosting", "#10B981")
    }

    m_key, m_title, m_theme_color = model_key_map[selected_diag_model]

    # Model Performance Snapshot Cards
    perf_metrics = {
        "linear": {"R²": "0.6583", "MAE": "$217.83", "RMSE": "$304.25", "MAPE": "16.07%", "Within ±10%": "43.2%", "Within ±20%": "71.1%"},
        "dt":     {"R²": "0.7352", "MAE": "$187.52", "RMSE": "$267.86", "MAPE": "14.16%", "Within ±10%": "50.2%", "Within ±20%": "77.4%"},
        "rf":     {"R²": "0.8318", "MAE": "$140.55", "RMSE": "$213.45", "MAPE": "10.60%", "Within ±10%": "64.6%", "Within ±20%": "87.1%"},
        "hgb":    {"R²": "0.8463", "MAE": "$140.07", "RMSE": "$204.04", "MAPE": "10.45%", "Within ±10%": "62.8%", "Within ±20%": "86.7%"}
    }

    cur_m = perf_metrics[m_key]
    pm1, pm2, pm3, pm4, pm5, pm6 = st.columns(6)
    pm1.markdown(f"<div class='metric-card'><div class='metric-label'>R² Score</div><div class='metric-val'>{cur_m['R²']}</div></div>", unsafe_allow_html=True)
    pm2.markdown(f"<div class='metric-card'><div class='metric-label'>MAE (USD)</div><div class='metric-val'>{cur_m['MAE']}</div></div>", unsafe_allow_html=True)
    pm3.markdown(f"<div class='metric-card'><div class='metric-label'>RMSE (USD)</div><div class='metric-val'>{cur_m['RMSE']}</div></div>", unsafe_allow_html=True)
    pm4.markdown(f"<div class='metric-card'><div class='metric-label'>MAPE</div><div class='metric-val'>{cur_m['MAPE']}</div></div>", unsafe_allow_html=True)
    pm5.markdown(f"<div class='metric-card'><div class='metric-label'>Within ±10%</div><div class='metric-val'>{cur_m['Within ±10%']}</div></div>", unsafe_allow_html=True)
    pm6.markdown(f"<div class='metric-card'><div class='metric-label'>Within ±20%</div><div class='metric-val'>{cur_m['Within ±20%']}</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if not df_diag.empty and f"pred_{m_key}" in df_diag.columns:
        diag_col1, diag_col2 = st.columns(2)

        with diag_col1:
            # Interactive Predicted vs Actual Plot
            fig_pva = go.Figure()
            # Identity diagonal line
            min_val = min(df_diag["price"].min(), df_diag[f"pred_{m_key}"].min())
            max_val = max(df_diag["price"].max(), df_diag[f"pred_{m_key}"].max())

            fig_pva.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(color="#EF4444", dash="dash", width=2),
                name="Perfect Prediction (y = x)"
            ))

            fig_pva.add_trace(go.Scatter(
                x=df_diag["price"],
                y=df_diag[f"pred_{m_key}"],
                mode='markers',
                marker=dict(
                    color=df_diag[f"abs_err_{m_key}"],
                    colorscale="Viridis",
                    size=5,
                    opacity=0.6,
                    colorbar=dict(title="Error ($)")
                ),
                name="Predictions",
                hovertemplate="Actual: <b>$%{x:,.0f}</b><br>Predicted: <b>$%{y:,.0f}</b><br>Error: <b>$%{marker.color:,.0f}</b><extra></extra>"
            ))
            fig_pva.update_layout(
                xaxis_title="Actual Monthly Rent ($ USD)",
                yaxis_title="Predicted Monthly Rent ($ USD)"
            )
            apply_plotly_styling(fig_pva, height=380, title=f"{m_title}: Predicted vs. Actual Rent")
            st.plotly_chart(fig_pva, use_container_width=True)

        with diag_col2:
            # Interactive Residual Plot (Residual vs Predicted)
            fig_res = go.Figure()
            fig_res.add_hline(y=0, line_dash="dash", line_color="#EF4444", line_width=2)
            fig_res.add_trace(go.Scatter(
                x=df_diag[f"pred_{m_key}"],
                y=df_diag[f"res_{m_key}"],
                mode='markers',
                marker=dict(color=m_theme_color, size=5, opacity=0.55),
                name="Residuals",
                hovertemplate="Predicted: <b>$%{x:,.0f}</b><br>Residual: <b>$%{y:,.0f}</b><extra></extra>"
            ))
            fig_res.update_layout(
                xaxis_title="Predicted Monthly Rent ($ USD)",
                yaxis_title="Residual (Actual − Predicted) in USD"
            )
            apply_plotly_styling(fig_res, height=380, title=f"{m_title}: Residual vs. Fitted Plot")
            st.plotly_chart(fig_res, use_container_width=True)

        diag_col3, diag_col4 = st.columns(2)
        with diag_col3:
            # Interactive Residual Histogram
            fig_res_hist = go.Figure()
            fig_res_hist.add_trace(go.Histogram(
                x=df_diag[f"res_{m_key}"],
                nbinsx=60,
                marker=dict(color=m_theme_color, line=dict(color="#FFFFFF", width=0.5)),
                name="Residual Distribution",
                hovertemplate="Residual: <b>$%{x:,.0f}</b><br>Frequency: <b>%{y:,}</b><extra></extra>"
            ))
            fig_res_hist.update_layout(
                xaxis_title="Residual Error ($ USD)",
                yaxis_title="Frequency",
                bargap=0.05
            )
            apply_plotly_styling(fig_res_hist, height=340, title=f"{m_title}: Residual Error Histogram")
            st.plotly_chart(fig_res_hist, use_container_width=True)

        with diag_col4:
            # Interactive Feature Importance (or Absolute Error by Tier)
            if m_key in ["dt", "rf", "hgb"]:
                importance_data = {
                    "dt": [
                        {"Feature": "City Median Price", "Importance": 36.2},
                        {"Feature": "Square Feet", "Importance": 18.5},
                        {"Feature": "Longitude", "Importance": 7.8},
                        {"Feature": "Latitude", "Importance": 6.4},
                        {"Feature": "Bathrooms", "Importance": 5.1},
                        {"Feature": "Sqft / Bedroom", "Importance": 4.8},
                        {"Feature": "Bedrooms", "Importance": 3.9},
                        {"Feature": "Sqft / Bathroom", "Importance": 3.1},
                        {"Feature": "Amenity Count", "Importance": 2.6},
                        {"Feature": "In-Unit Washer", "Importance": 1.8}
                    ],
                    "rf": [
                        {"Feature": "City Median Price", "Importance": 31.4},
                        {"Feature": "Square Feet", "Importance": 16.8},
                        {"Feature": "Longitude", "Importance": 9.2},
                        {"Feature": "Latitude", "Importance": 7.6},
                        {"Feature": "Bathrooms", "Importance": 5.8},
                        {"Feature": "Sqft / Bedroom", "Importance": 5.2},
                        {"Feature": "Bedrooms", "Importance": 4.3},
                        {"Feature": "Sqft / Bathroom", "Importance": 3.7},
                        {"Feature": "Amenity Count", "Importance": 3.1},
                        {"Feature": "In-Unit Washer", "Importance": 2.2}
                    ],
                    "hgb": [
                        {"Feature": "City Median Price", "Importance": 29.8},
                        {"Feature": "Square Feet", "Importance": 19.4},
                        {"Feature": "Longitude", "Importance": 10.1},
                        {"Feature": "Latitude", "Importance": 8.5},
                        {"Feature": "Bathrooms", "Importance": 6.2},
                        {"Feature": "Sqft / Bedroom", "Importance": 5.5},
                        {"Feature": "Sqft / Bathroom", "Importance": 4.1},
                        {"Feature": "Bedrooms", "Importance": 3.8},
                        {"Feature": "Amenity Count", "Importance": 2.9},
                        {"Feature": "In-Unit Washer", "Importance": 1.9}
                    ]
                }
                df_imp = pd.DataFrame(importance_data[m_key]).sort_values("Importance", ascending=True)
                fig_imp = px.bar(
                    df_imp,
                    x="Importance",
                    y="Feature",
                    orientation='h',
                    color="Importance",
                    color_continuous_scale="Blues",
                    text="Importance"
                )
                fig_imp.update_traces(
                    texttemplate="%{x:.1f}%",
                    textposition="outside",
                    textangle=0,
                    cliponaxis=False,
                    textfont=dict(color="#F8FAFC", size=11)
                )
                fig_imp.update_layout(
                    xaxis_title="Relative Feature Importance (%)",
                    yaxis_title="",
                    xaxis=dict(range=[0, df_imp["Importance"].max() * 1.18])
                )
                apply_plotly_styling(fig_imp, height=340, title=f"{m_title}: Top Feature Importances")
                st.plotly_chart(fig_imp, use_container_width=True)
            else:
                # For linear regression: MAE by Price Tier
                tier_df = df_diag.copy()
                tier_df["Price Tier"] = pd.cut(
                    tier_df["price"],
                    bins=[0, 1000, 2000, 3000, 5000],
                    labels=["<$1,000", "$1k–$2k", "$2k–$3k", "$3k+"]
                )
                tier_mae = tier_df.groupby("Price Tier", observed=False)[f"abs_err_{m_key}"].mean().reset_index()
                fig_tier = px.bar(
                    tier_mae,
                    x="Price Tier",
                    y=f"abs_err_{m_key}",
                    color="Price Tier",
                    color_discrete_sequence=["#3B82F6", "#10B981", "#F59E0B", "#EF4444"],
                    text=f"abs_err_{m_key}"
                )
                fig_tier.update_traces(
                    texttemplate="$%{y:.4f}",
                    textposition="auto",
                    textangle=0,
                    textfont=dict(size=12, color="#FFFFFF")
                )
                fig_tier.update_layout(xaxis_title="Rental Price Tier", yaxis_title="Mean Absolute Error ($ USD)", showlegend=False)
                apply_plotly_styling(fig_tier, height=340, title="Linear Regression: MAE Across Price Tiers")
                st.plotly_chart(fig_tier, use_container_width=True)


# =====================================================================
#  TAB 4 — EVALUATION & BENCHMARKS
# =====================================================================
with tab_eval:
    # Section 1: Interactive Benchmark Leaderboard
    st.markdown("### Model Leaderboard & Success Criteria")
    eval_table = pd.DataFrame([
        {
            "Rank": "1",
            "Model Architecture": "Hist Gradient Boosting",
            "R² Score": "0.8463",
            "MAE ($)": "$140.07",
            "RMSE ($)": "$204.04",
            "MAPE (%)": "10.45%",
            "±10% Accuracy": "62.8%",
            "±20% Accuracy": "86.7%"
        },
        {
            "Rank": "2",
            "Model Architecture": "Random Forest",
            "R² Score": "0.8318",
            "MAE ($)": "$140.55",
            "RMSE ($)": "$213.45",
            "MAPE (%)": "10.60%",
            "±10% Accuracy": "64.6%",
            "±20% Accuracy": "87.1%"
        },
        {
            "Rank": "3",
            "Model Architecture": "Decision Tree",
            "R² Score": "0.7352",
            "MAE ($)": "$187.52",
            "RMSE ($)": "$267.86",
            "MAPE (%)": "14.16%",
            "±10% Accuracy": "50.2%",
            "±20% Accuracy": "77.4%"
        },
        {
            "Rank": "4",
            "Model Architecture": "Linear Regression",
            "R² Score": "0.6583",
            "MAE ($)": "$217.83",
            "RMSE ($)": "$304.25",
            "MAPE (%)": "16.07%",
            "±10% Accuracy": "43.2%",
            "±20% Accuracy": "71.1%"
        }
    ])

    st.dataframe(
        eval_table,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Section 2: Interactive Comparative Visualizations
    st.markdown("<div class='section-header'><i class='bi bi-bar-chart-line'></i> Interactive Multi-Metric Performance Comparison</div>", unsafe_allow_html=True)

    comp_c1, comp_c2 = st.columns(2)
    models_list = ["Linear Reg.", "Decision Tree", "Random Forest", "Hist Grad Boost"]
    colors_list = ["#64748B", "#F59E0B", "#3B82F6", "#10B981"]

    with comp_c1:
        # R2 Comparison
        fig_r2 = go.Figure(go.Bar(
            x=models_list,
            y=[0.6583, 0.7352, 0.8318, 0.8463],
            marker=dict(color=colors_list),
            text=["0.6583", "0.7352", "0.8318", "<b>0.8463</b>"],
            textposition='auto',
            hovertemplate="Model: <b>%{x}</b><br>R² Score: <b>%{y:.4f}</b><extra></extra>"
        ))
        fig_r2.add_hline(y=0.80, line_dash="dash", line_color="#EF4444", annotation_text="Target (R² ≥ 0.80)")
        fig_r2.update_layout(yaxis_title="R² Score (Higher is Better)", yaxis_range=[0.5, 0.9])
        apply_plotly_styling(fig_r2, height=320, title="R² Variance Explained Benchmark")
        st.plotly_chart(fig_r2, use_container_width=True)

    with comp_c2:
        # MAE Comparison
        fig_mae = go.Figure(go.Bar(
            x=models_list,
            y=[217.83, 187.52, 140.55, 140.07],
            marker=dict(color=colors_list),
            text=["$218", "$188", "$141", "<b>$140</b>"],
            textposition='auto',
            hovertemplate="Model: <b>%{x}</b><br>MAE: <b>$%{y:.2f}</b><extra></extra>"
        ))
        fig_mae.add_hline(y=150.0, line_dash="dash", line_color="#EF4444", annotation_text="Target (MAE ≤ $150)")
        fig_mae.update_layout(yaxis_title="Mean Absolute Error ($ USD) (Lower is Better)")
        apply_plotly_styling(fig_mae, height=320, title="Mean Absolute Error (MAE) Benchmark")
        st.plotly_chart(fig_mae, use_container_width=True)

    comp_c3, comp_c4 = st.columns(2)
    with comp_c3:
        # RMSE Comparison
        fig_rmse = go.Figure(go.Bar(
            x=models_list,
            y=[304.25, 267.86, 213.45, 204.04],
            marker=dict(color=colors_list),
            text=["$304", "$268", "$213", "<b>$204</b>"],
            textposition='auto',
            hovertemplate="Model: <b>%{x}</b><br>RMSE: <b>$%{y:.2f}</b><extra></extra>"
        ))
        fig_rmse.update_layout(yaxis_title="Root Mean Squared Error ($ USD)")
        apply_plotly_styling(fig_rmse, height=320, title="Root Mean Squared Error (RMSE) Benchmark")
        st.plotly_chart(fig_rmse, use_container_width=True)

    with comp_c4:
        # Tolerance Accuracy
        fig_tol = go.Figure()
        fig_tol.add_trace(go.Bar(
            name="Within ±10%",
            x=models_list,
            y=[43.2, 50.2, 64.6, 62.8],
            marker=dict(color="#3B82F6"),
            text=["43.2%", "50.2%", "64.6%", "62.8%"],
            textposition='auto'
        ))
        fig_tol.add_trace(go.Bar(
            name="Within ±20%",
            x=models_list,
            y=[71.1, 77.4, 87.1, 86.7],
            marker=dict(color="#10B981"),
            text=["71.1%", "77.4%", "87.1%", "86.7%"],
            textposition='auto'
        ))
        fig_tol.add_hline(y=85.0, line_dash="dash", line_color="#EF4444", annotation_text="Target (±20% ≥ 85%)")
        fig_tol.update_layout(barmode='group', yaxis_title="Percentage of Test Listings (%)")
        apply_plotly_styling(fig_tol, height=320, title="Tolerance Band Accuracy (±10% and ±20%)")
        st.plotly_chart(fig_tol, use_container_width=True)

    st.markdown("---")

    # Section 3: 5-Fold Cross-Validation Verification
    st.markdown("<div class='section-header'><i class='bi bi-arrow-repeat'></i> 5-Fold Cross-Validation Generalization Verification</div>", unsafe_allow_html=True)
    cv_c1, cv_c2 = st.columns([3, 2])

    with cv_c1:
        cv_models = ["Linear Reg.", "Decision Tree", "Random Forest", "Hist Grad Boost"]
        cv_means = [0.6580, 0.7342, 0.8310, 0.8458]
        cv_stds = [0.0051, 0.0038, 0.0031, 0.0028]

        fig_cv = go.Figure()
        fig_cv.add_trace(go.Bar(
            x=cv_models,
            y=cv_means,
            error_y=dict(type='data', array=cv_stds, visible=True, color="#E2E8F0", thickness=2),
            marker=dict(color=colors_list),
            text=[f"{m:.4f} ± {s:.4f}" for m, s in zip(cv_means, cv_stds)],
            textposition='auto',
            hovertemplate="Model: <b>%{x}</b><br>Mean CV R²: <b>%{y:.4f}</b><br>Std Dev: <b>±%{error_y.array:.4f}</b><extra></extra>"
        ))
        fig_cv.update_layout(yaxis_title="5-Fold Mean CV R² Score", yaxis_range=[0.5, 0.9])
        apply_plotly_styling(fig_cv, height=330, title="5-Fold Cross-Validation Stability (Mean R² ± Std Dev)")
        st.plotly_chart(fig_cv, use_container_width=True)

    with cv_c2:
        st.markdown("**Cross-Validation Summary Table**")
        cv_table_df = pd.DataFrame({
            "Model": ["Linear Regression", "Decision Tree", "Random Forest", "Hist Gradient Boosting"],
            "Mean CV R²": [0.6580, 0.7342, 0.8310, 0.8458],
            "Std Dev": ["±0.0051", "±0.0038", "±0.0031", "±0.0028"],
            "Mean Log MAE": [0.1578, 0.1384, 0.1039, 0.1021]
        })
        st.dataframe(cv_table_df, use_container_width=True, hide_index=True)


# =====================================================================
#                              FOOTER
# =====================================================================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #94A3B8; font-size: 0.85rem; padding: 10px 0;'>"
    "US Apartment Rent Valuation Platform · Interactive Data Science & Machine Learning Suite · BMDS2003 Group 4"
    "</div>",
    unsafe_allow_html=True
)
