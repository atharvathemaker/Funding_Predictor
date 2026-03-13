"""
app.py
------
VC Funding Predictive Analytics Dashboard for Indian Tech Founders
Streamlit application — run with:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VC Funding Analytics — India",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 12px;
        color: white;
    }
    .metric-card h2 { font-size: 2rem; font-weight: 700; color: #e94560; margin: 0; }
    .metric-card p  { font-size: 0.85rem; color: #a0aec0; margin: 4px 0 0 0; }
    .insight-box {
        background: #f0f4ff;
        border-left: 4px solid #3b5bdb;
        border-radius: 4px;
        padding: 12px 16px;
        margin-top: 8px;
        font-size: 0.875rem;
        color: #2d3748;
    }
    .persona-badge {
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
        border-radius: 8px;
        padding: 14px 20px;
        font-size: 1.1rem;
        font-weight: 600;
        text-align: center;
        margin: 8px 0;
    }
    .stSidebar { background: #0f1117; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING  (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading trained models …")
def load_models():
    base = Path("models")
    return {
        "rf":        joblib.load(base / "rf_classifier.pkl"),
        "km":        joblib.load(base / "kmeans.pkl"),
        "reg":       joblib.load(base / "valuation_regressor.pkl"),
        "scaler":    joblib.load(base / "scaler.pkl"),
        "le_sector": joblib.load(base / "le_sector.pkl"),
        "le_demo":   joblib.load(base / "le_demo.pkl"),
        "le_status": joblib.load(base / "le_status.pkl"),
        "personas":  joblib.load(base / "persona_labels.pkl"),
    }


@st.cache_data(show_spinner="Loading market data …")
def load_data():
    return pd.read_csv("clean_startup_data.csv")


# ─────────────────────────────────────────────────────────────────────────────
# SAFE LABEL ENCODING (handle unseen categories)
# ─────────────────────────────────────────────────────────────────────────────
def safe_transform(le, value, default=0):
    try:
        return le.transform([value])[0]
    except ValueError:
        return default


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — Founder Inputs
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar(models):
    st.sidebar.image(
        "https://upload.wikimedia.org/wikipedia/en/4/41/Flag_of_India.svg",
        width=40,
    )
    st.sidebar.title("🔬 Startup Profiler")
    st.sidebar.caption("Enter your startup's details to generate predictions.")

    SECTORS = [
        "AI", "Fintech", "SaaS", "DeepTech", "Edtech",
        "Healthtech", "Agritech", "EV/CleanTech", "D2C/Consumer", "Logistics",
    ]
    DEMOS = ["India1", "India2", "Bharat", "Global"]

    sector   = st.sidebar.selectbox("Sector", SECTORS)
    demo     = st.sidebar.selectbox("Target Demographic", DEMOS)
    pedigree = st.sidebar.slider("Founder Pedigree Score", 1, 10, 6,
                                  help="IIT/IIM background, prior exits, tier-1 VC networks → 10")
    dpi      = st.sidebar.checkbox("DPI Leverage (Govt. support / PLI scheme)", value=False)
    arr      = st.sidebar.number_input("Annual Recurring Revenue (USD)", 0, 50_000_000,
                                        500_000, step=50_000, format="%d")
    cac      = st.sidebar.number_input("Customer Acquisition Cost (USD)", 0, 500_000,
                                        150, step=10, format="%d")
    burn     = st.sidebar.slider("Burn Multiple", 0.1, 10.0, 1.5, 0.1,
                                  help="Net Burn / Net New ARR — healthy < 2.0")
    ip       = st.sidebar.slider("IP / Moat Score", 1, 10, 5,
                                  help="Patents, proprietary data, regulatory moat → 10")

    st.sidebar.markdown("---")
    predict_btn = st.sidebar.button("⚡ Generate Predictions", type="primary",
                                     use_container_width=True)

    return dict(sector=sector, demo=demo, pedigree=pedigree, dpi=int(dpi),
                arr=arr, cac=cac, burn=burn, ip=ip), predict_btn


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def run_predictions(inputs: dict, models: dict):
    sec_enc  = safe_transform(models["le_sector"], inputs["sector"])
    demo_enc = safe_transform(models["le_demo"],   inputs["demo"])

    x_raw = np.array([[sec_enc, demo_enc, inputs["pedigree"], inputs["dpi"],
                        inputs["arr"], inputs["cac"], inputs["burn"], inputs["ip"]]])
    x_sc  = models["scaler"].transform(x_raw)

    # Classification – funding probability
    proba       = models["rf"].predict_proba(x_sc)[0]
    classes     = models["le_status"].classes_
    proba_dict  = dict(zip(classes, proba))
    funded_prob = proba_dict.get("Funded", 0.0)

    # Regression – valuation
    log_val   = models["reg"].predict(x_sc)[0]
    valuation = np.expm1(log_val)

    # Clustering – persona
    cluster     = models["km"].predict(x_sc)[0]
    persona     = models["personas"].get(cluster, f"Archetype {cluster}")

    return funded_prob, proba_dict, valuation, persona


# ─────────────────────────────────────────────────────────────────────────────
# GAUGE CHART
# ─────────────────────────────────────────────────────────────────────────────
def gauge_chart(probability: float) -> go.Figure:
    pct  = probability * 100
    if pct < 35:
        color = "#e74c3c"
    elif pct < 60:
        color = "#f39c12"
    else:
        color = "#2ecc71"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pct,
        number={"suffix": "%", "font": {"size": 36, "color": "white"}},
        delta={"reference": 45, "increasing": {"color": "#2ecc71"},
               "decreasing": {"color": "#e74c3c"}},
        title={"text": "Probability of Getting Funded", "font": {"size": 14, "color": "#a0aec0"}},
        gauge={
            "axis":  {"range": [0, 100], "tickcolor": "#a0aec0"},
            "bar":   {"color": color, "thickness": 0.25},
            "bgcolor": "rgba(0,0,0,0)",
            "steps": [
                {"range": [0,  35], "color": "rgba(231,76,60,0.15)"},
                {"range": [35, 60], "color": "rgba(243,156,18,0.15)"},
                {"range": [60, 100],"color": "rgba(46,204,113,0.15)"},
            ],
            "threshold": {"line": {"color": "white", "width": 3},
                          "thickness": 0.75, "value": pct},
        },
    ))
    fig.update_layout(
        height=280, margin=dict(t=40, b=10, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)", font_color="white",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# MARKET TREND CHARTS
# ─────────────────────────────────────────────────────────────────────────────
def chart_funding_by_sector(df: pd.DataFrame) -> go.Figure:
    funded = df[df["Funding_Status"] == "Funded"]
    agg    = (funded.groupby("Sector")["Total_Funding_USD"]
              .sum()
              .sort_values(ascending=True)
              .reset_index())
    agg["Total_M"] = agg["Total_Funding_USD"] / 1e6

    fig = px.bar(
        agg, x="Total_M", y="Sector", orientation="h",
        labels={"Total_M": "Total Funding (USD Millions)", "Sector": ""},
        color="Total_M",
        color_continuous_scale="Blues",
        title="💰 Total VC Funding by Sector (Funded Startups)",
    )
    fig.update_layout(
        coloraxis_showscale=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#2d3748",
        title_font_size=14,
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#edf2f7")
    fig.update_yaxes(showgrid=False)
    return fig


def chart_burn_vs_valuation(df: pd.DataFrame) -> go.Figure:
    sample = df.sample(min(800, len(df)), random_state=42)
    sample["Valuation_M"] = sample["Expected_Pre_Money_Valuation_USD"] / 1e6

    fig = px.scatter(
        sample,
        x="Burn_Multiple",
        y="Valuation_M",
        color="Funding_Status",
        size="IP_Moat_Score",
        hover_data=["Sector", "Founder_Pedigree_Score"],
        labels={"Burn_Multiple": "Burn Multiple",
                "Valuation_M": "Expected Pre-Money Valuation (USD M)"},
        color_discrete_map={"Funded": "#3b5bdb", "Bootstrapped": "#40c057",
                             "Dead": "#fa5252"},
        title="🔥 Burn Multiple vs. Expected Valuation",
        opacity=0.65,
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#2d3748",
        title_font_size=14,
        height=400,
        legend_title="Funding Status",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#edf2f7")
    fig.update_yaxes(showgrid=True, gridcolor="#edf2f7")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1a1a2e,#16213e);
                padding:28px 32px;border-radius:12px;margin-bottom:24px;">
        <h1 style="color:#e94560;margin:0;font-size:1.9rem;font-weight:700;">
            🇮🇳 VC Funding Predictive Analytics Dashboard
        </h1>
        <p style="color:#a0aec0;margin:8px 0 0 0;font-size:0.95rem;">
            AI-powered insights for Indian tech founders navigating the 2024–25 funding landscape.
            Enter your startup details in the sidebar and click <b>Generate Predictions</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load artifacts ────────────────────────────────────────────────────────
    try:
        models = load_models()
        df     = load_data()
    except FileNotFoundError as e:
        st.error(
            f"❌ Model or data file not found: **{e}**\n\n"
            "Please run the pipeline first:\n"
            "```\npython generate_data.py\npython train_models.py\n```"
        )
        st.stop()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    inputs, predict_btn = render_sidebar(models)

    # ── Prediction Panel ──────────────────────────────────────────────────────
    st.subheader("📊 Your Startup Predictions")

    if predict_btn:
        funded_prob, proba_dict, valuation, persona = run_predictions(inputs, models)

        col1, col2, col3 = st.columns([1.4, 1, 1])

        with col1:
            st.plotly_chart(gauge_chart(funded_prob), use_container_width=True)

        with col2:
            val_str = (f"${valuation/1e6:.1f}M" if valuation >= 1e6
                       else f"${valuation/1e3:.0f}K")
            st.markdown(f"""
            <div class="metric-card">
                <p>💎 Expected Pre-Money Valuation</p>
                <h2>{val_str}</h2>
                <p>Model: XGBoost / Linear Regression</p>
            </div>
            """, unsafe_allow_html=True)

            # Mini probability breakdown
            st.markdown("**Funding Outcome Probabilities**")
            for status, p in sorted(proba_dict.items(), key=lambda x: -x[1]):
                st.progress(float(p), text=f"{status}: {p*100:.1f}%")

        with col3:
            st.markdown("**🎯 Founder Persona / Archetype**")
            st.markdown(f'<div class="persona-badge">{persona}</div>',
                        unsafe_allow_html=True)

            st.markdown("**📋 Input Summary**")
            summary = pd.DataFrame({
                "Attribute": ["Sector", "ARR (USD)", "Burn Multiple",
                               "Pedigree", "IP Score", "DPI"],
                "Value": [inputs["sector"], f"${inputs['arr']:,}",
                           inputs["burn"], inputs["pedigree"],
                           inputs["ip"], "Yes" if inputs["dpi"] else "No"],
            })
            st.dataframe(summary, hide_index=True, use_container_width=True)
    else:
        st.info("👈 Fill in your startup details in the sidebar and click **Generate Predictions**.")

    # ── Market Trend Charts ───────────────────────────────────────────────────
    st.divider()
    st.subheader("🌏 Indian Startup Ecosystem — Market Intelligence")

    c1, c2 = st.columns(2)

    with c1:
        st.plotly_chart(chart_funding_by_sector(df), use_container_width=True)
        st.markdown("""
        <div class="insight-box">
        📌 <b>Insight 1:</b> Fintech and AI continue to command the lion's share of VC capital in India,
        reflecting global investor appetite for regulated-market disruption and LLM-era opportunities.
        Founders in these sectors benefit from more active term-sheet activity and higher revenue multiples.
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.plotly_chart(chart_burn_vs_valuation(df), use_container_width=True)
        st.markdown("""
        <div class="insight-box">
        📌 <b>Insight 2:</b> Startups with a Burn Multiple below 2.0 cluster at significantly higher
        pre-money valuations — validating the "profitability-first" thesis that has dominated Indian
        VC conversations since the 2022–23 funding winter. Efficient growth is now table-stakes for Series A+.
        </div>
        """, unsafe_allow_html=True)

    # ── Raw Data Explorer ─────────────────────────────────────────────────────
    with st.expander("🔍 Explore Clean Dataset"):
        sector_filter = st.multiselect(
            "Filter by Sector",
            options=sorted(df["Sector"].unique()),
            default=sorted(df["Sector"].unique())[:4],
        )
        status_filter = st.multiselect(
            "Filter by Funding Status",
            options=sorted(df["Funding_Status"].unique()),
            default=sorted(df["Funding_Status"].unique()),
        )
        filtered = df[
            df["Sector"].isin(sector_filter) &
            df["Funding_Status"].isin(status_filter)
        ]
        st.dataframe(
            filtered[["Startup_Name", "Sector", "Funding_Status",
                       "Revenue_ARR_USD", "Burn_Multiple",
                       "Expected_Pre_Money_Valuation_USD",
                       "Founder_Pedigree_Score", "IP_Moat_Score"]].head(200),
            use_container_width=True,
        )
        st.caption(f"Showing {min(200, len(filtered)):,} of {len(filtered):,} filtered rows.")

    # ── Footer ────────────────────────────────────────────────────────────────
    st.divider()
    st.caption(
        "⚠️ This dashboard uses **synthetic data** for illustrative purposes. "
        "Predictions are model outputs and should not constitute investment advice. "
        "Built with Streamlit · scikit-learn · XGBoost · Plotly"
    )


if __name__ == "__main__":
    main()
