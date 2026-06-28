"""
Phase 9 – Streamlit Dashboard
==============================
AgriPulse AI  |  Climate Change Impact on Global Food Supply
Interactive prediction, analytics and risk-assessment dashboard.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# pyrefly: ignore [missing-import]
import streamlit as st

# ── Page config (MUST be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="AgriPulse AI | Climate-Food Predictor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Global background ── */
.stApp { background: linear-gradient(135deg, #0f1117 0%, #1a1f2e 50%, #0f1117 100%); }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #161b2e 0%, #1e2540 100%);
    border-right: 1px solid rgba(99,179,237,0.15);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, rgba(99,179,237,0.08) 0%, rgba(159,122,234,0.08) 100%);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 16px;
    padding: 1rem 1.25rem;
    transition: transform 0.2s, box-shadow 0.2s;
}
[data-testid="metric-container"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(99,179,237,0.15);
}
[data-testid="metric-container"] label { color: #90cdf4 !important; font-size: 0.78rem !important; letter-spacing: 0.05em; text-transform: uppercase; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #e2e8f0 !important; font-weight: 700; font-size: 1.6rem !important; }
[data-testid="metric-container"] [data-testid="stMetricDelta"] { font-size: 0.78rem !important; }

/* ── Section headers ── */
h1 { color: #90cdf4 !important; font-weight: 700 !important; }
h2 { color: #bee3f8 !important; font-weight: 600 !important; }
h3 { color: #e2e8f0 !important; font-weight: 500 !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #4299e1 0%, #9f7aea 100%);
    color: white !important;
    border: none;
    border-radius: 12px;
    font-weight: 600;
    padding: 0.65rem 2rem;
    font-size: 1rem;
    transition: opacity 0.2s, transform 0.2s;
    width: 100%;
}
.stButton > button:hover {
    opacity: 0.9;
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(66,153,225,0.35);
}

/* ── Input widgets ── */
[data-baseweb="input"] input,
[data-baseweb="select"] div,
.stSlider .rc-slider { 
    background: rgba(255,255,255,0.04) !important;
    border-color: rgba(99,179,237,0.25) !important;
    color: #e2e8f0 !important;
    border-radius: 10px !important;
}
label[data-testid="stWidgetLabel"] p { color: #90cdf4 !important; font-size: 0.82rem !important; font-weight: 500 !important; letter-spacing: 0.03em; }

/* ── Info / success / warning boxes ── */
.stAlert { border-radius: 12px !important; }

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] { background: rgba(255,255,255,0.03); border-radius: 12px; padding: 4px; }
.stTabs [data-baseweb="tab"] { border-radius: 10px; color: #90cdf4 !important; font-weight: 500; }
.stTabs [aria-selected="true"] { background: rgba(66,153,225,0.2) !important; color: #e2e8f0 !important; }

/* ── Divider ── */
hr { border-color: rgba(99,179,237,0.15) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #1a1f2e; }
::-webkit-scrollbar-thumb { background: rgba(99,179,237,0.3); border-radius: 3px; }

/* ── Risk badge ── */
.risk-badge {
    display: inline-block;
    padding: 0.35rem 1rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.9rem;
    letter-spacing: 0.04em;
}
.risk-low    { background: rgba(72,187,120,0.2); color: #68d391; border: 1px solid rgba(72,187,120,0.4); }
.risk-medium { background: rgba(237,137,54,0.2); color: #f6ad55; border: 1px solid rgba(237,137,54,0.4); }
.risk-high   { background: rgba(245,101,101,0.2); color: #fc8181; border: 1px solid rgba(245,101,101,0.4); }

/* ── Hero gradient text ── */
.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #63b3ed, #9f7aea, #68d391);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.15;
}
.hero-sub { color: #a0aec0; font-size: 1.1rem; margin-top: 0.5rem; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# DATA & MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════

DATA_PATH      = "data/processed/engineered_data.csv"
PROCESSED_PATH = "data/processed/processed_data.csv"
IMPORTANCE_PATH = "output/feature_importance_baseline.csv"

COUNTRIES = ['Bangladesh', 'Brazil', 'China', 'India', 'Indonesia',
             'Mexico', 'Nigeria', 'Pakistan', 'Russia', 'USA']
CROPS     = ['maize', 'rice', 'soybean', 'wheat']
CROP_EMOJI = {'maize': '🌽', 'rice': '🍚', 'soybean': '🫘', 'wheat': '🌾'}


@st.cache_data(show_spinner=False)
def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_importance():
    if os.path.exists(IMPORTANCE_PATH):
        return pd.read_csv(IMPORTANCE_PATH)
    return pd.DataFrame()


@st.cache_resource(show_spinner=False)
def load_models():
    """Load saved models if available; return empty dict otherwise."""
    # pyrefly: ignore [missing-import]
    import joblib
    loaded = {}
    model_files = {
        "XGBoost":          "models/xgboost.pkl",
        "Random Forest":    "models/random_forest.pkl",
        "Gradient Boosting":"models/gradient_boosting.pkl",
    }
    feature_path = "models/feature_cols.pkl"
    for name, path in model_files.items():
        if os.path.exists(path):
            try:
                loaded[name] = joblib.load(path)
            except Exception:
                pass
    feature_cols = joblib.load(feature_path) if os.path.exists(feature_path) else []
    return loaded, feature_cols


df          = load_data()
importance  = load_importance()
models_dict, feature_cols = load_models()
models_available = len(models_dict) > 0


# ═══════════════════════════════════════════════════════════════════════════
# PLOTLY THEME HELPER
# ═══════════════════════════════════════════════════════════════════════════

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.03)",
    font=dict(family="Inter", color="#e2e8f0"),
    title_font=dict(family="Inter", color="#90cdf4", size=16),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(99,179,237,0.2)"),
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.08)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.08)"),
    margin=dict(t=50, b=40, l=40, r=20),
)

COLOR_SEQ = ["#63b3ed", "#9f7aea", "#68d391", "#f6ad55", "#fc8181",
             "#76e4f7", "#fbb6ce", "#b794f4", "#9ae6b4", "#fed7aa"]


def apply_theme(fig):
    fig.update_layout(**PLOT_LAYOUT)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# RISK SCORING
# ═══════════════════════════════════════════════════════════════════════════

def compute_risk(temperature, rainfall, co2, humidity):
    """Compute 0-100 food security risk score from raw climate inputs."""
    # Temperature component: penalise deviation from 20°C optimum
    temp_risk  = min(1.0, abs(temperature - 20) / 25)
    # Rainfall component: penalise deviation from 550 mm optimum
    rain_risk  = min(1.0, abs(rainfall - 550) / 550)
    # CO₂ component: risk rises above 400 ppm
    co2_risk   = min(1.0, max(0, (co2 - 400) / 200))
    # Humidity component: risk outside 50-80 % band
    hum_risk   = min(1.0, abs(humidity - 65) / 65)

    score = (0.40 * temp_risk + 0.30 * rain_risk +
             0.20 * co2_risk  + 0.10 * hum_risk) * 100
    return round(score, 1)


def risk_label(score):
    if score < 33:
        return "🟢 Low Risk", "risk-low"
    elif score < 66:
        return "🟡 Moderate Risk", "risk-medium"
    else:
        return "🔴 High Risk", "risk-high"


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🌍 AgriPulse AI")
    st.markdown("<hr>", unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["🏠  Home", "📊  Dashboard", "🔮  Predictions", "📈  Analytics", "ℹ️  About"],
        label_visibility="collapsed",
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("**Model Status**")
    if models_available:
        for name in models_dict:
            st.success(f"✔ {name}")
    else:
        st.warning("No trained models found.\nRun `model_trainer.py` first.")

    st.markdown("<hr>", unsafe_allow_html=True)
    if not df.empty:
        st.caption(f"📦 Dataset: **{len(df):,}** records")
        st.caption(f"📅 Years: **{int(df['Year'].min())}–{int(df['Year'].max())}**")
        st.caption(f"🌾 Crops: **{', '.join(CROPS)}**")
        st.caption(f"🌐 Countries: **{df['Country'].nunique()}**")
    st.markdown("<hr>", unsafe_allow_html=True)
    st.caption("Phase 9 · AgriPulse AI v1.0")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ═══════════════════════════════════════════════════════════════════════════

if page == "🏠  Home":
    # Hero
    st.markdown("""
    <div style="padding: 2.5rem 0 1.5rem 0;">
        <div class="hero-title">Climate Change Impact<br>on Global Food Supply</div>
        <div class="hero-sub">AI-powered crop yield forecasting & food security risk assessment</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Stats row
    if not df.empty:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg Temperature",   f"{df['Temperature'].mean():.1f} °C",   f"↑ {df.groupby('Year')['Temperature'].mean().iloc[-1] - df.groupby('Year')['Temperature'].mean().iloc[0]:.2f} °C since 2000")
        c2.metric("Avg Rainfall",      f"{df['Rainfall'].mean():.0f} mm",      f"{df.groupby('Year')['Rainfall'].mean().iloc[-1] - df.groupby('Year')['Rainfall'].mean().iloc[0]:+.0f} mm trend")
        c3.metric("Avg CO₂",           f"{df['CO2_Emission'].mean():.0f} ppm", f"↑ {df['CO2_Emission'].max() - df['CO2_Emission'].min():.0f} ppm range")
        c4.metric("Avg Crop Yield",    f"{df['Yield'].mean():.1f} kg/ha",      f"σ = {df['Yield'].std():.1f}")

    st.markdown("---")

    # Feature cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style="background:rgba(99,179,237,0.07);border:1px solid rgba(99,179,237,0.2);border-radius:16px;padding:1.4rem 1.6rem">
            <h3>🔮 Yield Predictions</h3>
            <p style="color:#a0aec0;font-size:0.9rem">Input real-time climate variables and get instant crop yield predictions across 4 crops and 10 countries — powered by XGBoost, Random Forest & ensemble models.</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="background:rgba(159,122,234,0.07);border:1px solid rgba(159,122,234,0.2);border-radius:16px;padding:1.4rem 1.6rem">
            <h3>⚠️ Risk Assessment</h3>
            <p style="color:#a0aec0;font-size:0.9rem">Quantify food security risk from 0–100 using a composite score that combines temperature stress, rainfall deficit, CO₂ levels and humidity anomalies.</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style="background:rgba(104,211,145,0.07);border:1px solid rgba(104,211,145,0.2);border-radius:16px;padding:1.4rem 1.6rem">
            <h3>📈 Deep Analytics</h3>
            <p style="color:#a0aec0;font-size:0.9rem">Explore climate trends, feature importance rankings, crop-specific correlations, and country-level yield comparisons with interactive Plotly charts.</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Crop & variable reference
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🌾 Supported Crops")
        for crop in CROPS:
            st.markdown(f"- {CROP_EMOJI[crop]} **{crop.title()}**")
    with c2:
        st.subheader("🌡️ Climate Variables")
        st.markdown("""
        - 🌡️ **Temperature** — Annual mean (°C)
        - 🌧️ **Rainfall** — Annual precipitation (mm)
        - 💨 **CO₂ Emissions** — Concentration (ppm)
        - 💧 **Humidity** — Relative humidity (%)
        """)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════

elif page == "📊  Dashboard":
    st.header("📊 Data Dashboard")

    if df.empty:
        st.warning("No data found at `data/processed/engineered_data.csv`. Run the pipeline first.")
        st.stop()

    # ── Filters ──────────────────────────────────────────────────────────
    with st.expander("🔧 Filter Data", expanded=False):
        fc1, fc2, fc3 = st.columns(3)
        sel_countries = fc1.multiselect("Countries", COUNTRIES, default=COUNTRIES[:5], key="dash_countries")
        sel_crops     = fc2.multiselect("Crops",     CROPS,     default=CROPS,         key="dash_crops")
        year_min, year_max = int(df['Year'].min()), int(df['Year'].max())
        sel_years = fc3.slider("Year Range", year_min, year_max, (year_min, year_max), key="dash_years")

    filtered = df.copy()
    if sel_countries: filtered = filtered[filtered['Country'].isin(sel_countries)]
    if sel_crops:     filtered = filtered[filtered['Crop'].isin(sel_crops)]
    filtered = filtered[(filtered['Year'] >= sel_years[0]) & (filtered['Year'] <= sel_years[1])]

    st.caption(f"Showing **{len(filtered):,}** records")
    st.markdown("---")

    # ── KPIs ─────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Countries",   filtered['Country'].nunique())
    k2.metric("Crops",       filtered['Crop'].nunique())
    k3.metric("Avg Temp",    f"{filtered['Temperature'].mean():.1f} °C")
    k4.metric("Avg Rainfall",f"{filtered['Rainfall'].mean():.0f} mm")
    k5.metric("Avg Yield",   f"{filtered['Yield'].mean():.1f} kg/ha")

    st.markdown("---")

    # ── Tabs ─────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["🌡️ Climate Trends", "🌾 Yield Analysis", "🗺️ Country Comparison", "🔗 Correlations"])

    with tab1:
        yearly = filtered.groupby('Year').agg(
            Temperature=('Temperature','mean'),
            Rainfall=('Rainfall','mean'),
            CO2_Emission=('CO2_Emission','mean'),
            Yield=('Yield','mean'),
        ).reset_index()

        fig = make_subplots(rows=2, cols=2,
                            subplot_titles=["Temperature (°C)", "Rainfall (mm)",
                                            "CO₂ Emissions (ppm)", "Avg Crop Yield (kg/ha)"],
                            vertical_spacing=0.14, horizontal_spacing=0.1)

        for (row, col, col_name, color) in [
            (1, 1, "Temperature",  "#63b3ed"),
            (1, 2, "Rainfall",     "#68d391"),
            (2, 1, "CO2_Emission", "#f6ad55"),
            (2, 2, "Yield",        "#9f7aea"),
        ]:
            fig.add_trace(go.Scatter(
                x=yearly['Year'], y=yearly[col_name],
                mode='lines+markers',
                name=col_name.replace("_", " "),
                line=dict(color=color, width=2.5),
                marker=dict(size=5),
                showlegend=False,
            ), row=row, col=col)

        fig.update_layout(height=520, **PLOT_LAYOUT, title_text="Climate & Yield Trends Over Time")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        cy1, cy2 = st.columns(2)

        # Box plot: yield by crop
        fig_box = px.box(filtered, x='Crop', y='Yield', color='Crop',
                         color_discrete_sequence=COLOR_SEQ,
                         title="Yield Distribution by Crop",
                         labels={'Yield': 'Yield (kg/ha)'})
        apply_theme(fig_box)
        cy1.plotly_chart(fig_box, use_container_width=True)

        # Line: avg yield by year per crop
        crop_yearly = filtered.groupby(['Year','Crop'])['Yield'].mean().reset_index()
        fig_line = px.line(crop_yearly, x='Year', y='Yield', color='Crop',
                           color_discrete_sequence=COLOR_SEQ,
                           title="Avg Yield per Year by Crop",
                           labels={'Yield': 'Yield (kg/ha)'})
        apply_theme(fig_line)
        cy2.plotly_chart(fig_line, use_container_width=True)

        # Violin: temperature vs yield
        fig_vio = px.violin(filtered, x='Crop', y='Yield', color='Crop',
                            color_discrete_sequence=COLOR_SEQ,
                            box=True, points=False,
                            title="Yield Violin Plot by Crop")
        apply_theme(fig_vio)
        st.plotly_chart(fig_vio, use_container_width=True)

    with tab3:
        country_stats = filtered.groupby('Country').agg(
            Avg_Yield=('Yield','mean'),
            Avg_Temp=('Temperature','mean'),
            Avg_Rain=('Rainfall','mean'),
            Records=('Yield','count'),
        ).reset_index().sort_values('Avg_Yield', ascending=False)

        fig_bar = px.bar(country_stats, x='Country', y='Avg_Yield',
                         color='Avg_Temp', color_continuous_scale='RdYlBu_r',
                         title="Average Yield by Country (colour = avg temperature)",
                         labels={'Avg_Yield': 'Avg Yield (kg/ha)', 'Avg_Temp': '°C'})
        apply_theme(fig_bar)
        st.plotly_chart(fig_bar, use_container_width=True)

        fig_scat = px.scatter(country_stats, x='Avg_Rain', y='Avg_Yield',
                              size='Records', color='Country',
                              color_discrete_sequence=COLOR_SEQ,
                              hover_name='Country',
                              title="Country: Rainfall vs Yield (bubble = record count)",
                              labels={'Avg_Rain':'Avg Rainfall (mm)', 'Avg_Yield':'Avg Yield (kg/ha)'})
        apply_theme(fig_scat)
        st.plotly_chart(fig_scat, use_container_width=True)

    with tab4:
        num_cols = ['Temperature','Rainfall','CO2_Emission','Humidity','Yield',
                    'Climate_Stress_Index','Growing_Season_Quality','Extreme_Weather_Risk']
        avail = [c for c in num_cols if c in filtered.columns]
        corr = filtered[avail].corr()

        fig_heat = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.index,
            colorscale='RdBu', zmin=-1, zmax=1,
            text=np.round(corr.values, 2),
            texttemplate='%{text}',
            hoverongaps=False,
        ))
        fig_heat.update_layout(title="Feature Correlation Matrix", height=520, **PLOT_LAYOUT)
        st.plotly_chart(fig_heat, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════

elif page == "🔮  Predictions":
    st.header("🔮 Yield Prediction & Risk Assessment")

    if df.empty:
        st.warning("Load data first by running the pipeline.")
        st.stop()

    st.markdown("Set climate conditions below and get an instant prediction.")

    # ── Input form ───────────────────────────────────────────────────────
    with st.form("prediction_form"):
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("🌡️ Climate Variables")
            temperature = st.slider("Temperature (°C)", -10.0, 50.0, 22.0, 0.5, key="p_temp")
            rainfall    = st.slider("Rainfall (mm)",      0.0, 2000.0, 600.0, 10.0, key="p_rain")
            co2         = st.slider("CO₂ Emissions (ppm)", 300.0, 580.0, 420.0, 1.0, key="p_co2")
            humidity    = st.slider("Humidity (%)",         0.0, 100.0,  65.0, 1.0, key="p_hum")

        with c2:
            st.subheader("🌾 Crop & Location")
            crop    = st.selectbox("Crop",    [c.title() for c in CROPS],     key="p_crop")
            country = st.selectbox("Country", COUNTRIES,                       key="p_country")
            year    = st.number_input("Year", min_value=2024, max_value=2050,
                                      value=2025, key="p_year")
            model_choice = st.selectbox(
                "Prediction Model",
                list(models_dict.keys()) + (["📊 Statistical Estimate"] if not models_available else []),
                key="p_model"
            )
            st.markdown("")
            submitted = st.form_submit_button("🎯 Predict Yield", use_container_width=True)

    if submitted:
        crop_lower = crop.lower()

        # ── Statistical fallback prediction ──────────────────────────────
        if not models_available or model_choice == "📊 Statistical Estimate":
            # Use dataset median for country-crop as baseline
            mask = (df['Country'] == country) & (df['Crop'] == crop_lower)
            base_yield = df.loc[mask, 'Yield'].median() if mask.sum() > 0 else df['Yield'].median()

            # Climate adjustments (simple linear factors)
            temp_adj  = -(temperature - 20) * 0.8          # penalty per °C above 20
            rain_adj  =  (rainfall - 550)   * 0.01         # bonus per mm above 550
            co2_adj   =  (co2 - 350)        * 0.02         # small fertilisation benefit
            hum_adj   = -(abs(humidity-65)) * 0.05         # penalty for humidity extremes

            raw_pred = base_yield + temp_adj + rain_adj + co2_adj + hum_adj
            prediction = max(5.0, raw_pred)
            model_label = "Statistical Estimate"

        else:
            # ── ML model prediction ──────────────────────────────────────
            # pyrefly: ignore [missing-import]
            import joblib
            model_obj = models_dict[model_choice]

            # Build a one-row input that mirrors the engineered features
            min_year = int(df['Year'].min())
            country_baseline = df.groupby('Country')['Yield'].mean().get(country, df['Yield'].mean())
            co2_mean = df['CO2_Emission'].mean()

            # Compute engineered features the same way feature_engineer.py does
            temp_norm = (temperature - df['Temperature'].mean()) / df['Temperature'].std()
            rain_norm = (rainfall    - df['Rainfall'].mean())    / df['Rainfall'].std()
            co2_norm  = (co2         - co2_mean)                 / df['CO2_Emission'].std()

            climate_stress = 0.45*temp_norm + 0.30*(-rain_norm) + 0.25*co2_norm
            temp_rain_ratio = temperature / (rainfall + 1)
            ideal_temp  = np.clip(1 - abs(temperature - 20) / 20, 0, 1)
            ideal_rain  = np.clip(1 - abs(rainfall - 550) / 550, 0, 1)
            growing_q   = 0.6*ideal_temp + 0.4*ideal_rain
            ext_risk    = 0.6*np.clip(abs(temperature-20)/20, 0, 1) + 0.4*np.clip(abs(rainfall-550)/550, 0, 1)
            co2_anomaly = co2 - co2_mean
            co2_fert    = np.log(co2 / 280) * 5
            clim_co2    = climate_stress * co2_anomaly
            years_since = year - min_year
            decade      = (year // 10) * 10
            clim_chg    = years_since * 0.03

            row = {f: 0.0 for f in feature_cols}

            # Fill in known engineered values
            for key, val in {
                'Temperature': temperature, 'Rainfall': rainfall,
                'CO2_Emission': co2, 'Humidity': humidity,
                'Climate_Stress_Index': climate_stress,
                'Temp_Rain_Ratio': temp_rain_ratio,
                'Growing_Season_Quality': growing_q,
                'Extreme_Weather_Risk': ext_risk,
                'Years_Since_Baseline': years_since,
                'Decade': decade,
                'Climate_Change_Index': clim_chg,
                'CO2_Anomaly': co2_anomaly,
                'CO2_Fertilization_Effect': co2_fert,
                'Climate_CO2_Interaction': clim_co2,
                'Country_Yield_Baseline': country_baseline,
                'Yield_vs_Baseline': 0.0,
                'Country_Yield_Trend': 0.0,
            }.items():
                if key in row:
                    row[key] = val

            # Crop interaction columns
            for c_name in CROPS:
                flag = 1.0 if c_name == crop_lower else 0.0
                if f'Crop_{c_name}_Temp' in row: row[f'Crop_{c_name}_Temp'] = flag * temperature
                if f'Crop_{c_name}_Rain' in row: row[f'Crop_{c_name}_Rain'] = flag * rainfall

            X_input = pd.DataFrame([row])[feature_cols]
            prediction = float(model_obj.predict(X_input)[0])
            model_label = model_choice

        # ── Results ──────────────────────────────────────────────────────
        risk_score   = compute_risk(temperature, rainfall, co2, humidity)
        rlabel, rcls = risk_label(risk_score)

        st.markdown("---")
        st.subheader("📋 Prediction Results")

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("🌾 Predicted Yield",  f"{prediction:.1f} kg/ha")
        r2.metric("⚠️ Risk Score",        f"{risk_score:.0f} / 100")
        r3.metric("📐 Model",             model_label)

        # Context comparison
        mask = (df['Crop'] == crop_lower)
        historical_avg = df.loc[mask, 'Yield'].mean()
        delta_pct = (prediction - historical_avg) / historical_avg * 100
        r4.metric("📊 vs Historical Avg",
                  f"{historical_avg:.1f} kg/ha",
                  f"{delta_pct:+.1f}%")

        # Risk badge
        st.markdown(f"""
        <div style="margin:1rem 0">
            <span class="risk-badge {rcls}">{rlabel}</span>
        </div>""", unsafe_allow_html=True)

        # Risk breakdown gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            title={"text": "Food Security Risk", "font": {"color": "#90cdf4", "size": 16}},
            number={"suffix": "/100", "font": {"color": "#e2e8f0", "size": 36}},
            gauge={
                "axis":  {"range": [0, 100], "tickcolor": "#e2e8f0"},
                "bar":   {"color": "#fc8181" if risk_score > 65 else ("#f6ad55" if risk_score > 32 else "#68d391")},
                "bgcolor": "rgba(255,255,255,0.05)",
                "bordercolor": "rgba(99,179,237,0.2)",
                "steps": [
                    {"range": [0, 33],  "color": "rgba(104,211,145,0.12)"},
                    {"range": [33, 66], "color": "rgba(246,173,85,0.12)"},
                    {"range": [66, 100],"color": "rgba(252,129,129,0.12)"},
                ],
                "threshold": {"line": {"color": "white", "width": 3}, "value": risk_score},
            }
        ))
        fig_gauge.update_layout(height=280, **PLOT_LAYOUT)
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Climate input radar
        cats   = ['Temperature', 'Rainfall', 'CO₂', 'Humidity']
        norms  = [
            min(1, abs(temperature - 20) / 30),
            min(1, abs(rainfall - 550)   / 700),
            min(1, (co2 - 300) / 280),
            min(1, abs(humidity - 65)    / 65),
        ]
        fig_radar = go.Figure(go.Scatterpolar(
            r=norms + [norms[0]],
            theta=cats + [cats[0]],
            fill='toself',
            fillcolor='rgba(99,179,237,0.15)',
            line=dict(color='#63b3ed', width=2),
            name='Stress Level',
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor='rgba(255,255,255,0.03)',
                radialaxis=dict(visible=True, range=[0, 1], color='#90cdf4'),
                angularaxis=dict(color='#e2e8f0'),
            ),
            title="Climate Stress Radar",
            height=380, **PLOT_LAYOUT,
        )
        st.plotly_chart(fig_radar, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════

elif page == "📈  Analytics":
    st.header("📈 Advanced Analytics")

    if df.empty:
        st.warning("Load data first.")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["🏆 Feature Importance", "🌡️ Climate–Yield", "🔮 Scenario Explorer"])

    # ── Tab 1: Feature importance ─────────────────────────────────────────
    with tab1:
        if not importance.empty:
            top_n = st.slider("Show top N features", 5, min(30, len(importance)), 15, key="fn")
            top = importance.head(top_n)

            fig_imp = go.Figure(go.Bar(
                y=top['Feature'],
                x=top['Abs_Correlation'],
                orientation='h',
                marker=dict(
                    color=top['Abs_Correlation'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Importance"),
                ),
                text=[f"{v:.3f}" for v in top['Abs_Correlation']],
                textposition='outside',
            ))
            fig_imp.update_layout(
                title=f"Top {top_n} Features by Correlation with Yield",
                xaxis_title="Absolute Correlation",
                height=max(400, top_n * 28),
                **PLOT_LAYOUT,
            )
            fig_imp.update_yaxes(autorange='reversed')
            st.plotly_chart(fig_imp, use_container_width=True)

            with st.expander("📄 Full Feature Table"):
                st.dataframe(
                    importance.style.background_gradient(subset=['Abs_Correlation'], cmap='Blues'),
                    use_container_width=True,
                )
        else:
            st.info("Feature importance file not found. Run `feature_engineer.py` to generate it.")

    # ── Tab 2: Climate–Yield scatter ──────────────────────────────────────
    with tab2:
        a1, a2 = st.columns(2)
        x_var = a1.selectbox("X axis", ['Temperature','Rainfall','CO2_Emission','Humidity',
                                         'Climate_Stress_Index','Growing_Season_Quality'], key="ax")
        color_var = a2.selectbox("Colour by", ['Crop','Country'], key="ac")

        fig_sc = px.scatter(df, x=x_var, y='Yield', color=color_var,
            color_discrete_sequence=COLOR_SEQ,
            opacity=0.65,
            title=f"{x_var.replace('_',' ')} vs Yield",
            labels={'Yield': 'Yield (kg/ha)'})
        apply_theme(fig_sc)
        st.plotly_chart(fig_sc, use_container_width=True)

        # Per-crop correlation table
        corr_rows = []
        for crop_name in CROPS:
            sub = df[df['Crop'] == crop_name]
            for col in ['Temperature','Rainfall','CO2_Emission','Humidity']:
                if col in sub.columns:
                    corr_rows.append({
                        'Crop': crop_name.title(),
                        'Variable': col,
                        'Correlation': round(sub[col].corr(sub['Yield']), 3),
                    })
        corr_df = pd.DataFrame(corr_rows)
        pivot   = corr_df.pivot(index='Variable', columns='Crop', values='Correlation')

        fig_corr = go.Figure(go.Heatmap(
            z=pivot.values, x=pivot.columns, y=pivot.index,
            colorscale='RdBu', zmin=-1, zmax=1,
            text=np.round(pivot.values, 2), texttemplate='%{text}',
        ))
        fig_corr.update_layout(title="Crop-specific Climate–Yield Correlations", height=320, **PLOT_LAYOUT)
        st.plotly_chart(fig_corr, use_container_width=True)

    # ── Tab 3: Scenario explorer ──────────────────────────────────────────
    with tab3:
        st.markdown("Sweep a single climate variable and see how predicted yield changes.")

        sc1, sc2, sc3 = st.columns(3)
        sweep_var = sc1.selectbox("Variable to sweep", ['Temperature','Rainfall','CO2_Emission','Humidity'], key="sv")
        sweep_crop = sc2.selectbox("Crop", [c.title() for c in CROPS], key="sc")
        sweep_country = sc3.selectbox("Country", COUNTRIES, key="sco")

        # Fixed baselines
        base_temp = 20.0; base_rain = 600.0; base_co2 = 420.0; base_hum = 65.0

        sweep_ranges = {
            'Temperature':  np.linspace(-5,  45,  60),
            'Rainfall':     np.linspace(0,  2000, 60),
            'CO2_Emission': np.linspace(300, 580, 60),
            'Humidity':     np.linspace(0,  100,  60),
        }
        xs = sweep_ranges[sweep_var]

        # Statistical estimate for sweep
        mask = (df['Country'] == sweep_country) & (df['Crop'] == sweep_crop.lower())
        base_yield = df.loc[mask, 'Yield'].median() if mask.sum() > 0 else df['Yield'].median()

        preds = []
        for x in xs:
            t = x if sweep_var == 'Temperature' else base_temp
            r = x if sweep_var == 'Rainfall'    else base_rain
            c = x if sweep_var == 'CO2_Emission' else base_co2
            h = x if sweep_var == 'Humidity'    else base_hum
            ta = -(t - 20) * 0.8
            ra =  (r - 550) * 0.01
            ca =  (c - 350) * 0.02
            ha = -(abs(h - 65)) * 0.05
            preds.append(max(5.0, base_yield + ta + ra + ca + ha))

        risks = [compute_risk(
            x if sweep_var=='Temperature' else base_temp,
            x if sweep_var=='Rainfall'    else base_rain,
            x if sweep_var=='CO2_Emission' else base_co2,
            x if sweep_var=='Humidity'    else base_hum,
        ) for x in xs]

        fig_sweep = make_subplots(specs=[[{"secondary_y": True}]])
        fig_sweep.add_trace(go.Scatter(x=xs, y=preds,  name="Predicted Yield (kg/ha)",
                                       line=dict(color="#63b3ed", width=2.5)), secondary_y=False)
        fig_sweep.add_trace(go.Scatter(x=xs, y=risks,  name="Risk Score (0-100)",
                                       line=dict(color="#fc8181", width=2.5, dash='dash')), secondary_y=True)
        fig_sweep.update_xaxes(title_text=sweep_var.replace('_',' '))
        fig_sweep.update_yaxes(title_text="Yield (kg/ha)",   secondary_y=False, color="#63b3ed")
        fig_sweep.update_yaxes(title_text="Risk Score",      secondary_y=True,  color="#fc8181")
        fig_sweep.update_layout(title=f"Scenario: Yield & Risk vs {sweep_var.replace('_',' ')} "
                                      f"({sweep_crop}, {sweep_country})",
                                height=430, **PLOT_LAYOUT)
        st.plotly_chart(fig_sweep, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ═══════════════════════════════════════════════════════════════════════════

elif page == "ℹ️  About":
    st.header("ℹ️ About AgriPulse AI")

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("""
        ### 🎯 Project Objective
        AgriPulse AI predicts the impact of climate change on global crop production using
        machine learning models trained on historical climate and agricultural data spanning
        **2000–2023** across **10 countries** and **4 major crops**.

        ### 🗄️ Data Sources
        | Source | Data | Coverage |
        |--------|------|----------|
        | FAO STAT | Crop production, yield, area | 1961–2023 |
        | World Bank | Temperature, rainfall | 1960–2023 |
        | NASA GISS | Temperature anomalies | 1880–2024 |
        | NOAA | Precipitation patterns | 1850–2024 |

        ### 🤖 Models Used
        | Model | Strength |
        |-------|---------|
        | **XGBoost** | High accuracy, handles non-linearity |
        | **Random Forest** | Robust, good feature importance |
        | **Gradient Boosting** | Balanced bias-variance tradeoff |
        | **Ensemble** | Weighted average for best overall |

        ### 🔍 Explainability
        - **SHAP** (SHapley Additive exPlanations) — feature attribution
        - **Feature Correlation Baseline** — pre-model importance ranking
        """)

    with c2:
        st.markdown("""
        ### 📦 Pipeline Phases
        """)
        phases = [
            ("✅", "Phase 1", "Setup & Environment"),
            ("✅", "Phase 2", "Data Collection"),
            ("✅", "Phase 3", "EDA"),
            ("✅", "Phase 4", "Data Cleaning"),
            ("✅", "Phase 5", "Feature Engineering"),
            ("✅", "Phase 6", "Model Training"),
            ("✅", "Phase 7", "Optimization"),
            ("✅", "Phase 8", "Explainable AI"),
            ("✅", "Phase 9", "Streamlit Dashboard"),
            ("🟡", "Phase 10", "Future Predictions"),
            ("🟡", "Phase 11", "Testing"),
            ("⬜", "Phase 12", "Deployment"),
        ]
        for icon, phase, desc in phases:
            st.markdown(f"{icon} **{phase}** — {desc}")

        st.markdown("---")
        st.markdown("""
        **Version:** 1.0  
        **Built with:** Python · Streamlit · Plotly  
        **ML Stack:** scikit-learn · XGBoost · SHAP  
        """)

    st.markdown("---")
    st.info("💡 **Tip:** Run the full pipeline before using ML predictions — "
            "`python src/data_processing/data_cleaner.py` → "
            "`feature_engineer.py` → `model_trainer.py`")
