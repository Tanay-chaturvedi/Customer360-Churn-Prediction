# =============================================================================
# Customer360 – Customer Retention Intelligence Platform
# Enterprise SaaS Analytics Dashboard (Streamlit Frontend)
# =============================================================================

import os
import sys
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from sqlalchemy import text

# Configure structured logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("customer360_dashboard")

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database.db_connection import get_engine

# =============================================================================
# PAGE CONFIGURATION & ENTERPRISE DESIGN SYSTEM
# =============================================================================
st.set_page_config(
    page_title="Customer360 | Retention Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

ENTERPRISE_CSS = """
<style>
    /* ---------------- Global Typography & Palette ---------------- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: #0F172A;
    }

    .stApp {
        background-color: #F8FAFC;
    }

    /* ---------------- Main Content Layout & Top Header Clearance ---------------- */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    .main .block-container,
    [data-testid="stMainBlockContainer"],
    section.main > div.block-container {
        padding-top: 4.25rem !important;
        padding-bottom: 3rem;
        max-width: 1240px;
    }

    /* ---------------- Headings & Text ---------------- */
    h1, h2, h3, h4, h5, h6 {
        color: #0F172A !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }

    .dashboard-header {
        margin-top: 0.25rem;
        margin-bottom: 1.25rem;
        padding-top: 0.25rem;
    }
    .dashboard-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0F172A;
        line-height: 1.3;
        margin: 0 0 6px 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .dashboard-subtitle {
        font-size: 0.88rem;
        color: #64748B;
        margin: 0;
        line-height: 1.4;
    }

    /* ---------------- KPI Metric Cards ---------------- */
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 18px 20px;
        box-shadow: 0 1px 3px 0 rgba(15, 23, 42, 0.04), 0 1px 2px -1px rgba(15, 23, 42, 0.02);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        height: 100%;
        position: relative;
        overflow: hidden;
    }
    .kpi-card:hover {
        box-shadow: 0 4px 6px -1px rgba(15, 23, 42, 0.07), 0 2px 4px -2px rgba(15, 23, 42, 0.04);
    }
    .kpi-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: #E2E8F0;
    }
    .kpi-card.accent-blue::before { background: #2563EB; }
    .kpi-card.accent-red::before { background: #DC2626; }
    .kpi-card.accent-amber::before { background: #D97706; }
    .kpi-card.accent-green::before { background: #16A34A; }

    .kpi-label {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0F172A;
        line-height: 1.15;
        margin: 0 0 6px 0;
        letter-spacing: -0.02em;
    }
    .kpi-desc {
        font-size: 0.78rem;
        color: #64748B;
        margin: 0;
        font-weight: 500;
    }

    /* ---------------- Chart Cards ---------------- */
    .chart-container {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px 18px 12px 18px;
        box-shadow: 0 1px 3px 0 rgba(15, 23, 42, 0.03);
        margin-bottom: 16px;
    }
    .chart-header {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        margin-bottom: 10px;
        padding-bottom: 8px;
        border-bottom: 1px solid #F1F5F9;
    }
    .chart-title {
        font-size: 0.92rem;
        font-weight: 700;
        color: #0F172A;
        margin: 0;
    }
    .chart-tag {
        font-size: 0.72rem;
        color: #64748B;
        font-weight: 500;
    }

    /* ---------------- Analytical Callout Banners ---------------- */
    .insight-callout {
        background-color: #F8FAFC;
        border-left: 3px solid #2563EB;
        border-radius: 0 6px 6px 0;
        padding: 9px 14px;
        font-size: 0.82rem;
        color: #334155;
        margin: 6px 0 14px 0;
        line-height: 1.45;
        border-top: 1px solid #F1F5F9;
        border-right: 1px solid #F1F5F9;
        border-bottom: 1px solid #F1F5F9;
    }
    .insight-callout strong {
        color: #0F172A;
    }

    .advisory-box {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 3px solid #3B82F6;
        padding: 12px 16px;
        border-radius: 8px;
        font-size: 0.82rem;
        color: #475569;
        margin: 10px 0 18px 0;
        line-height: 1.5;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
    }
    .advisory-box strong {
        color: #0F172A;
    }

    /* ---------------- Risk Badges ---------------- */
    .badge-high {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 3px 9px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.76rem;
        border: 1px solid #FCA5A5;
        display: inline-flex;
        align-items: center;
        gap: 4px;
    }
    .badge-medium {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 3px 9px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.76rem;
        border: 1px solid #FCD34D;
        display: inline-flex;
        align-items: center;
        gap: 4px;
    }
    .badge-low {
        background-color: #DCFCE7;
        color: #166534;
        padding: 3px 9px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.76rem;
        border: 1px solid #86EFAC;
        display: inline-flex;
        align-items: center;
        gap: 4px;
    }

    /* ---------------- Playbook Cards ---------------- */
    .playbook-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 18px 20px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);
        height: 100%;
        border-top: 3px solid #2563EB;
    }
    .playbook-card.p-high { border-top-color: #DC2626; }
    .playbook-card.p-med { border-top-color: #D97706; }
    .playbook-card.p-info { border-top-color: #2563EB; }

    .playbook-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 10px;
    }
    .playbook-title {
        font-size: 0.98rem;
        font-weight: 700;
        color: #0F172A;
        margin: 0;
    }
    .cohort-badge {
        background-color: #F1F5F9;
        color: #334155;
        padding: 3px 8px;
        border-radius: 5px;
        font-size: 0.72rem;
        font-weight: 600;
        border: 1px solid #E2E8F0;
    }
    .playbook-section-label {
        font-size: 0.72rem;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-top: 10px;
        margin-bottom: 3px;
    }
    .playbook-body {
        font-size: 0.84rem;
        color: #334155;
        line-height: 1.45;
    }

    /* ---------------- Transparency Cards ---------------- */
    .info-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px 18px;
        font-size: 0.82rem;
        color: #334155;
        height: 100%;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.02);
    }
    .info-card-title {
        font-size: 0.78rem;
        font-weight: 700;
        color: #0F172A;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 10px;
        border-bottom: 1px solid #F1F5F9;
        padding-bottom: 6px;
    }

    /* ---------------- Sidebar Styling ---------------- */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }
    [data-testid="stSidebar"] .block-container,
    [data-testid="stSidebar"] section > div {
        padding-top: 2rem !important;
    }

    .sidebar-brand-box {
        display: flex;
        flex-direction: column;
        gap: 3px;
        padding-bottom: 14px;
        margin-bottom: 14px;
        border-bottom: 1px solid #E2E8F0;
    }
    .sidebar-logo-text {
        font-size: 1.25rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.03em;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .sidebar-subtitle-text {
        font-size: 0.74rem;
        color: #64748B;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .api-status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        padding: 6px 10px;
        border-radius: 6px;
        font-size: 0.74rem;
        color: #334155;
        font-weight: 500;
        width: 100%;
    }
    .status-dot-active {
        width: 8px;
        height: 8px;
        background-color: #10B981;
        border-radius: 50%;
        display: inline-block;
    }
    .status-dot-offline {
        width: 8px;
        height: 8px;
        background-color: #EF4444;
        border-radius: 50%;
        display: inline-block;
    }

    /* Streamlit Tabs & Buttons */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #E2E8F0;
        padding-bottom: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.88rem;
        font-weight: 600;
        color: #64748B;
        padding: 8px 16px;
        border-radius: 6px;
    }
    .stTabs [aria-selected="true"] {
        color: #2563EB !important;
        background-color: #EFF6FF !important;
    }
</style>
"""
st.markdown(ENTERPRISE_CSS, unsafe_allow_html=True)


# =============================================================================
# FASTAPI INTEGRATION CONFIGURATION
# =============================================================================
API_BASE_URL: str = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
_API_TIMEOUT: int = 30  # seconds


def _check_api_health() -> Optional[dict]:
    """Return the /health payload if the API is reachable, else None."""
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except requests.exceptions.RequestException:
        pass
    return None


# =============================================================================
# DATA ACCESS (MySQL direct with CSV fallback; inference via FastAPI)
# =============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def load_customer_data() -> pd.DataFrame:
    """Fetch customer records from MySQL database with automatic CSV fallback.

    Historical analytics (churn rates, charges, tenure) are computed directly
    from this DataFrame. Only ML inference is delegated to FastAPI.
    """
    try:
        engine = get_engine(include_db=True)
        with engine.connect() as conn:
            df = pd.read_sql_query(text("SELECT * FROM `customers`;"), conn)
            df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
            df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").astype(int)
            logger.info(f"Loaded {len(df):,} records from MySQL.")
            return df
    except Exception as e:
        logger.warning(f"MySQL unavailable ({e}). Falling back to data/raw/customer_churn.csv.")
        csv_path = PROJECT_ROOT / "data" / "raw" / "customer_churn.csv"
        df = pd.read_csv(csv_path)
        df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
        df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").astype(int)
        return df


@st.cache_data(show_spinner=False, ttl=3600)
def score_customers_via_api(df_input: pd.DataFrame, chunk_size: int = 5000) -> pd.DataFrame:
    """Send customer records to the FastAPI /predict/batch endpoint in chunks of
    up to 5,000 records and return the original DataFrame enriched with
    churn_probability, churn_probability_pct, predicted_churn, and risk_category columns.
    """
    if df_input.empty:
        scored_df = df_input.copy()
        scored_df["churn_probability"] = pd.Series(dtype=float)
        scored_df["churn_probability_pct"] = pd.Series(dtype=float)
        scored_df["predicted_churn"] = pd.Series(dtype=str)
        scored_df["risk_category"] = pd.Series(dtype=str)
        return scored_df

    # Exclude Churn label from feature payload but keep customerID for traceability
    keep_cols = [c for c in df_input.columns if c not in ("Churn",)]
    feature_df = df_input[keep_cols].copy()

    records = feature_df.to_dict(orient="records")
    all_predictions = []

    # Send records in sequential chunks of up to chunk_size (FastAPI limit is 5,000)
    for start_idx in range(0, len(records), chunk_size):
        chunk = records[start_idx : start_idx + chunk_size]
        payload = {"customers": chunk}

        try:
            resp = requests.post(
                f"{API_BASE_URL}/predict/batch",
                json=payload,
                timeout=_API_TIMEOUT,
            )
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Cannot reach FastAPI service at {API_BASE_URL}. "
                "Please start the API service."
            )
        except requests.exceptions.Timeout:
            raise RuntimeError(
                f"FastAPI request timed out after {_API_TIMEOUT}s for batch chunk "
                f"[{start_idx}:{start_idx + len(chunk)}]."
            )

        if resp.status_code != 200:
            raise RuntimeError(
                f"FastAPI /predict/batch returned HTTP {resp.status_code}: {resp.text[:400]}"
            )

        chunk_predictions = resp.json()["predictions"]
        if len(chunk_predictions) != len(chunk):
            raise RuntimeError(
                f"FastAPI returned {len(chunk_predictions)} predictions for a chunk of size {len(chunk)}."
            )
        all_predictions.extend(chunk_predictions)

    if len(all_predictions) != len(df_input):
        raise RuntimeError(
            f"Expected {len(df_input)} total predictions, but received {len(all_predictions)} from API."
        )

    pred_df = pd.DataFrame(all_predictions)
    scored_df = df_input.copy()
    scored_df["churn_probability"] = pred_df["churn_probability"].values
    scored_df["churn_probability_pct"] = np.round(pred_df["churn_probability"].values * 100, 1)
    scored_df["predicted_churn"] = pred_df["predicted_churn"].values
    scored_df["risk_category"] = pred_df["risk_category"].values
    logger.info("Received %d total predictions across chunks from FastAPI.", len(scored_df))
    return scored_df


def predict_single_customer_via_api(customer_dict: dict) -> dict:
    """Send a single customer's features to FastAPI /predict and return result."""
    try:
        resp = requests.post(
            f"{API_BASE_URL}/predict",
            json=customer_dict,
            timeout=_API_TIMEOUT,
        )
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Cannot reach FastAPI service at {API_BASE_URL}. Please ensure API is active."
        )
    except requests.exceptions.Timeout:
        raise RuntimeError(f"FastAPI request timed out after {_API_TIMEOUT}s.")

    if resp.status_code != 200:
        raise RuntimeError(
            f"FastAPI /predict returned HTTP {resp.status_code}: {resp.text[:400]}"
        )

    return resp.json()


# =============================================================================
# REUSABLE ENTERPRISE PLOTLY THEME
# =============================================================================
def apply_enterprise_theme(fig: go.Figure, height: int = 310) -> go.Figure:
    """Apply consistent enterprise SaaS styling to Plotly figures."""
    fig.update_layout(
        font=dict(
            family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif",
            color="#334155",
            size=12
        ),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        height=height,
        margin=dict(l=30, r=20, t=20, b=25),
        xaxis=dict(
            showline=True,
            linecolor="#E2E8F0",
            linewidth=1,
            gridcolor="#F8FAFC",
            tickfont=dict(color="#64748B", size=11),
            title_font=dict(color="#1E293B", size=12, weight=600)
        ),
        yaxis=dict(
            showline=False,
            gridcolor="#F1F5F9",
            gridwidth=1,
            tickfont=dict(color="#64748B", size=11),
            title_font=dict(color="#1E293B", size=12, weight=600)
        ),
        hoverlabel=dict(
            bgcolor="#0F172A",
            font_size=12,
            font_family="Inter",
            font_color="#FFFFFF"
        )
    )
    return fig


# =============================================================================
# REFINED NAVIGATION SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand-box">
        <div class="sidebar-logo-text">
            <span style="color: #2563EB;">⬢</span> Customer360
        </div>
        <div class="sidebar-subtitle-text">Customer Retention Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    section = st.radio(
        "Navigation",
        [
            "Business Overview",
            "Churn Analysis",
            "Customer Risk Prediction",
            "Retention Insights"
        ],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("<div style='height: 120px;'></div>", unsafe_allow_html=True)
    st.markdown("---")

    # Compact API Connection Status Badge
    _health = _check_api_health()
    if _health and _health.get("model_loaded"):
        st.markdown(f"""
        <div class="api-status-badge">
            <span class="status-dot-active"></span>
            <div>
                <strong>FastAPI Connected</strong><br>
                <span style="color: #64748B; font-size: 0.70rem;">XGBoost Champion • v1.0</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="api-status-badge">
            <span class="status-dot-offline"></span>
            <div>
                <strong style="color: #DC2626;">API Service Offline</strong><br>
                <span style="color: #64748B; font-size: 0.70rem;">Historical analytics only</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# DATA INITIALIZATION & BATCH PREDICTION
# =============================================================================
try:
    raw_df = load_customer_data()
except Exception as err:
    st.error(f"Data load error: {err}")
    st.stop()

try:
    scored_customers_df = score_customers_via_api(raw_df)
except RuntimeError as _api_err:
    st.warning(
        f"FastAPI microservice is not responding at {API_BASE_URL}. "
        "Business Overview and Churn Analysis remain fully operational."
    )
    scored_customers_df = raw_df.copy()
    scored_customers_df["churn_probability"] = float("nan")
    scored_customers_df["churn_probability_pct"] = float("nan")
    scored_customers_df["predicted_churn"] = "Unknown"
    scored_customers_df["risk_category"] = "Unknown"
except Exception as err:
    st.error(f"Unexpected scoring error: {err}")
    st.stop()


# =============================================================================
# 1. SECTION A: BUSINESS OVERVIEW
# =============================================================================
if section == "Business Overview":
    st.markdown("""
    <div class="dashboard-header">
        <div class="dashboard-title">Business Overview</div>
        <div class="dashboard-subtitle">Enterprise customer health, historical attrition baseline, and recurring revenue exposure.</div>
    </div>
    """, unsafe_allow_html=True)

    # Key Performance Indicators
    total_customers = len(raw_df)
    churned_customers = int((raw_df["Churn"] == "Yes").sum())
    retained_customers = int((raw_df["Churn"] == "No").sum())
    churn_rate = (churned_customers / total_customers) * 100
    churned_mrr = float(raw_df[raw_df["Churn"] == "Yes"]["MonthlyCharges"].sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""
        <div class="kpi-card accent-blue">
            <div class="kpi-label">Total Customers</div>
            <div class="kpi-value">{total_customers:,}</div>
            <div class="kpi-desc">Active account portfolio</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kpi-card accent-red">
            <div class="kpi-label">Churned Customers</div>
            <div class="kpi-value" style="color: #DC2626;">{churned_customers:,}</div>
            <div class="kpi-desc">Historical departures</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="kpi-card accent-amber">
            <div class="kpi-label">Overall Churn Rate</div>
            <div class="kpi-value" style="color: #D97706;">{churn_rate:.1f}%</div>
            <div class="kpi-desc">Baseline attrition rate</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="kpi-card accent-green">
            <div class="kpi-label">Retained Customers</div>
            <div class="kpi-value" style="color: #16A34A;">{retained_customers:,}</div>
            <div class="kpi-desc">73.5% sustained accounts</div>
        </div>
        """, unsafe_allow_html=True)
    with c5:
        st.markdown(f"""
        <div class="kpi-card accent-red">
            <div class="kpi-label">Monthly Revenue at Risk</div>
            <div class="kpi-value" style="color: #991B1B;">${churned_mrr:,.0f}</div>
            <div class="kpi-desc">Monthly recurring loss</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Chart Grid 1: Distribution & Monthly Charges
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("""
        <div class="chart-header">
            <span class="chart-title">Customer Retention vs. Churn Distribution</span>
            <span class="chart-tag">Baseline Composition</span>
        </div>
        """, unsafe_allow_html=True)

        churn_counts = raw_df["Churn"].value_counts()
        fig_donut = px.pie(
            values=churn_counts.values,
            names=["Retained", "Churned"],
            hole=0.62,
            color=["Retained", "Churned"],
            color_discrete_map={"Retained": "#2563EB", "Churned": "#DC2626"}
        )
        fig_donut.update_traces(
            textposition="outside",
            textinfo="percent+label",
            marker=dict(line=dict(color="#FFFFFF", width=2)),
            hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Proportion: %{percent}<extra></extra>"
        )
        fig_donut.update_layout(
            showlegend=False,
            annotations=[dict(
                text=f"<b>{churn_rate:.1f}%</b><br><span style='font-size:11px;color:#64748B;'>Churn</span>",
                x=0.5, y=0.5, font_size=17, font_family="Inter", showarrow=False
            )],
            margin=dict(l=20, r=20, t=10, b=10),
            height=290,
            paper_bgcolor="#FFFFFF"
        )
        st.plotly_chart(fig_donut, width="stretch")

    with col_right:
        st.markdown("""
        <div class="chart-header">
            <span class="chart-title">Monthly Charges Distribution ($)</span>
            <span class="chart-tag">Median & Variance</span>
        </div>
        """, unsafe_allow_html=True)

        fig_charges = px.box(
            raw_df,
            x="Churn",
            y="MonthlyCharges",
            color="Churn",
            color_discrete_map={"No": "#2563EB", "Yes": "#DC2626"},
            points="outliers",
            category_orders={"Churn": ["No", "Yes"]}
        )
        apply_enterprise_theme(fig_charges, height=290)
        fig_charges.update_layout(
            xaxis_title="Historical Churn Status",
            yaxis_title="Monthly Bill ($)",
            showlegend=False,
            xaxis=dict(ticktext=["Retained (No)", "Churned (Yes)"], tickvals=["No", "Yes"])
        )
        st.plotly_chart(fig_charges, width="stretch")

    # Chart Row 2: Customer Tenure Comparison
    st.markdown("""
    <div class="chart-header" style="margin-top: 10px;">
        <span class="chart-title">Customer Tenure Comparison: Churned vs. Retained (Months)</span>
        <span class="chart-tag">Lifecycle Duration</span>
    </div>
    """, unsafe_allow_html=True)

    fig_tenure = px.histogram(
        raw_df,
        x="tenure",
        color="Churn",
        barmode="overlay",
        marginal="box",
        nbins=36,
        color_discrete_map={"No": "#2563EB", "Yes": "#DC2626"},
        category_orders={"Churn": ["No", "Yes"]}
    )
    apply_enterprise_theme(fig_tenure, height=320)
    fig_tenure.update_layout(
        xaxis_title="Tenure (Months with Company)",
        yaxis_title="Customer Count",
        legend=dict(
            title="",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color="#475569")
        )
    )
    st.plotly_chart(fig_tenure, width="stretch")


# =============================================================================
# 2. SECTION B: CHURN ANALYSIS
# =============================================================================
elif section == "Churn Analysis":
    st.markdown("""
    <div class="dashboard-header">
        <div class="dashboard-title">Churn Analysis</div>
        <div class="dashboard-subtitle">Multi-dimensional exploration of attrition drivers across contracts, payment methods, and services.</div>
    </div>
    """, unsafe_allow_html=True)

    # Filter Controls Bar
    with st.container():
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            contract_options = ["All Contracts"] + list(raw_df["Contract"].unique())
            selected_contract = st.selectbox("Contract Type", contract_options, index=0)
        with f_col2:
            internet_options = ["All Internet Services"] + list(raw_df["InternetService"].unique())
            selected_internet = st.selectbox("Internet Service", internet_options, index=0)
        with f_col3:
            min_tenure, max_tenure = int(raw_df["tenure"].min()), int(raw_df["tenure"].max())
            tenure_range = st.slider("Tenure Range (Months)", min_tenure, max_tenure, (min_tenure, max_tenure))

    # Apply filters
    filtered_df = raw_df.copy()
    if selected_contract != "All Contracts":
        filtered_df = filtered_df[filtered_df["Contract"] == selected_contract]
    if selected_internet != "All Internet Services":
        filtered_df = filtered_df[filtered_df["InternetService"] == selected_internet]
    filtered_df = filtered_df[
        (filtered_df["tenure"] >= tenure_range[0]) & (filtered_df["tenure"] <= tenure_range[1])
    ]

    st.caption(f"Analyzing **{len(filtered_df):,}** matching customer accounts (out of {len(raw_df):,})")

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # 1. Contract & Lifecycle Dynamics Section
    st.markdown("#### 1. Contract Terms & Tenure Cohort Dynamics")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="chart-header">
            <span class="chart-title">Attrition Rate by Contract Type</span>
            <span class="chart-tag">Structural Commitment</span>
        </div>
        """, unsafe_allow_html=True)

        contract_summary = (
            filtered_df.groupby("Contract", observed=False)["Churn"]
            .apply(lambda s: (s == "Yes").mean() * 100 if len(s) > 0 else 0)
            .reset_index(name="Churn_Rate_Pct")
        )
        fig_contract = px.bar(
            contract_summary,
            x="Contract",
            y="Churn_Rate_Pct",
            color="Contract",
            color_discrete_map={
                "Month-to-month": "#DC2626",
                "One year": "#2563EB",
                "Two year": "#16A34A"
            },
            text=contract_summary["Churn_Rate_Pct"].apply(lambda v: f"{v:.1f}%")
        )
        apply_enterprise_theme(fig_contract, height=270)
        fig_contract.update_layout(
            yaxis_title="Churn Rate (%)",
            xaxis_title="",
            showlegend=False
        )
        st.plotly_chart(fig_contract, width="stretch")
        st.markdown("""
        <div class="insight-callout">
            <strong>Key Finding:</strong> Month-to-month accounts exhibit <strong>42.7% churn</strong>, compared to under 3% for two-year commitments.
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="chart-header">
            <span class="chart-title">Attrition Rate by Tenure Cohort</span>
            <span class="chart-tag">Account Aging</span>
        </div>
        """, unsafe_allow_html=True)

        bins = [-1, 6, 12, 24, 48, 72]
        labels = ["0-6 Mos", "7-12 Mos", "13-24 Mos", "25-48 Mos", "49-72 Mos"]
        filtered_df["Tenure_Group"] = pd.cut(filtered_df["tenure"], bins=bins, labels=labels)

        tenure_summary = (
            filtered_df.groupby("Tenure_Group", observed=False)["Churn"]
            .apply(lambda s: (s == "Yes").mean() * 100 if len(s) > 0 else 0)
            .reset_index(name="Churn_Rate_Pct")
        )
        fig_tenure_grp = px.bar(
            tenure_summary,
            x="Tenure_Group",
            y="Churn_Rate_Pct",
            color="Churn_Rate_Pct",
            color_continuous_scale="Blues",
            text=tenure_summary["Churn_Rate_Pct"].apply(lambda v: f"{v:.1f}%")
        )
        apply_enterprise_theme(fig_tenure_grp, height=270)
        fig_tenure_grp.update_layout(
            yaxis_title="Churn Rate (%)",
            xaxis_title="",
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_tenure_grp, width="stretch")
        st.markdown("""
        <div class="insight-callout">
            <strong>Key Finding:</strong> Churn risk is heavily front-loaded during the first 12 months (51.9% in the 0–1 month cohort).
        </div>
        """, unsafe_allow_html=True)

    # 2. Service & Payment Method Dynamics Section
    st.markdown("#### 2. Service Offering & Payment Method Dynamics")
    c3, c4 = st.columns(2)

    with c3:
        st.markdown("""
        <div class="chart-header">
            <span class="chart-title">Attrition Rate by Internet Service</span>
            <span class="chart-tag">Network Architecture</span>
        </div>
        """, unsafe_allow_html=True)

        internet_summary = (
            filtered_df.groupby("InternetService", observed=False)["Churn"]
            .apply(lambda s: (s == "Yes").mean() * 100 if len(s) > 0 else 0)
            .reset_index(name="Churn_Rate_Pct")
        )
        fig_internet = px.bar(
            internet_summary,
            x="InternetService",
            y="Churn_Rate_Pct",
            color="InternetService",
            color_discrete_map={
                "Fiber optic": "#DC2626",
                "DSL": "#2563EB",
                "No": "#16A34A"
            },
            text=internet_summary["Churn_Rate_Pct"].apply(lambda v: f"{v:.1f}%")
        )
        apply_enterprise_theme(fig_internet, height=270)
        fig_internet.update_layout(
            yaxis_title="Churn Rate (%)",
            xaxis_title="",
            showlegend=False
        )
        st.plotly_chart(fig_internet, width="stretch")
        st.markdown("""
        <div class="insight-callout">
            <strong>Key Finding:</strong> Fiber optic subscribers suffer <strong>41.9% churn</strong>, signaling potential pricing friction or early support issues.
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown("""
        <div class="chart-header">
            <span class="chart-title">Attrition Rate by Payment Method</span>
            <span class="chart-tag">Billing Friction</span>
        </div>
        """, unsafe_allow_html=True)

        payment_summary = (
            filtered_df.groupby("PaymentMethod", observed=False)["Churn"]
            .apply(lambda s: (s == "Yes").mean() * 100 if len(s) > 0 else 0)
            .reset_index(name="Churn_Rate_Pct")
            .sort_values(by="Churn_Rate_Pct", ascending=False)
        )
        fig_payment = px.bar(
            payment_summary,
            x="PaymentMethod",
            y="Churn_Rate_Pct",
            color="PaymentMethod",
            color_discrete_map={
                "Electronic check": "#DC2626",
                "Mailed check": "#F59E0B",
                "Bank transfer (automatic)": "#2563EB",
                "Credit card (automatic)": "#16A34A"
            },
            text=payment_summary["Churn_Rate_Pct"].apply(lambda v: f"{v:.1f}%")
        )
        apply_enterprise_theme(fig_payment, height=270)
        fig_payment.update_layout(
            yaxis_title="Churn Rate (%)",
            xaxis_title="",
            showlegend=False
        )
        st.plotly_chart(fig_payment, width="stretch")
        st.markdown("""
        <div class="insight-callout">
            <strong>Key Finding:</strong> Electronic check users churn at <strong>45.3%</strong>, compared to ~15-16% for automated credit card or bank debit channels.
        </div>
        """, unsafe_allow_html=True)

    # 3. Monthly Spend Distribution
    st.markdown("#### 3. Monthly Spend Exposure & Density")
    fig_mc_dist = px.histogram(
        filtered_df,
        x="MonthlyCharges",
        color="Churn",
        barmode="overlay",
        nbins=40,
        color_discrete_map={"No": "#2563EB", "Yes": "#DC2626"},
        category_orders={"Churn": ["No", "Yes"]}
    )
    apply_enterprise_theme(fig_mc_dist, height=280)
    fig_mc_dist.update_layout(
        xaxis_title="Monthly Charges ($)",
        yaxis_title="Customer Count",
        legend=dict(
            title="", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(size=11, color="#475569")
        )
    )
    st.plotly_chart(fig_mc_dist, width="stretch")


# =============================================================================
# 3. SECTION C: CUSTOMER RISK PREDICTION
# =============================================================================
elif section == "Customer Risk Prediction":
    st.markdown("""
    <div class="dashboard-header">
        <div class="dashboard-title">Customer Risk Prediction</div>
        <div class="dashboard-subtitle">Real-time model inference and customer risk scoring via the champion XGBoost pipeline.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="advisory-box">
        <strong>Governance & Methodological Standard:</strong>
        Predictions are generated strictly using operational attributes without <code>customerID</code> or target labels entering model inputs.
        Risk tiers are business operational classifications (<strong>High Risk &ge; 0.70</strong>, <strong>Medium Risk 0.40–0.69</strong>, <strong>Low Risk &lt; 0.40</strong>) serving proactive intervention prioritization.
    </div>
    """, unsafe_allow_html=True)

    tab_portfolio, tab_predict_new = st.tabs(["Customer Risk Portfolio", "Predict New Customer"])

    # ---------------- TAB 1: PORTFOLIO ----------------
    with tab_portfolio:
        total_scored = len(scored_customers_df)
        high_risk_count = int((scored_customers_df["risk_category"] == "High Risk").sum())
        med_risk_count = int((scored_customers_df["risk_category"] == "Medium Risk").sum())
        low_risk_count = int((scored_customers_df["risk_category"] == "Low Risk").sum())
        high_risk_mrr = float(
            scored_customers_df[scored_customers_df["risk_category"] == "High Risk"]["MonthlyCharges"].sum()
        )

        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.markdown(f"""
            <div class="kpi-card accent-red">
                <div class="kpi-label">High Risk Tier (&ge;0.70)</div>
                <div class="kpi-value" style="color: #DC2626;">{high_risk_count:,}</div>
                <div class="kpi-desc">{high_risk_count/total_scored*100:.1f}% of total portfolio</div>
            </div>
            """, unsafe_allow_html=True)
        with r2:
            st.markdown(f"""
            <div class="kpi-card accent-amber">
                <div class="kpi-label">Medium Risk (0.40 - 0.69)</div>
                <div class="kpi-value" style="color: #D97706;">{med_risk_count:,}</div>
                <div class="kpi-desc">{med_risk_count/total_scored*100:.1f}% of total portfolio</div>
            </div>
            """, unsafe_allow_html=True)
        with r3:
            st.markdown(f"""
            <div class="kpi-card accent-green">
                <div class="kpi-label">Low Risk (&lt;0.40)</div>
                <div class="kpi-value" style="color: #16A34A;">{low_risk_count:,}</div>
                <div class="kpi-desc">{low_risk_count/total_scored*100:.1f}% of total portfolio</div>
            </div>
            """, unsafe_allow_html=True)
        with r4:
            st.markdown(f"""
            <div class="kpi-card accent-red">
                <div class="kpi-label">High-Risk Monthly MRR</div>
                <div class="kpi-value" style="color: #991B1B;">${high_risk_mrr:,.0f}</div>
                <div class="kpi-desc">Monthly revenue exposure</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

        # Filters & Search Controls
        t_col1, t_col2, t_col3, t_col4 = st.columns([1.5, 1.5, 1.5, 2.0])
        with t_col1:
            risk_filter = st.selectbox("Filter Risk Category", ["All Categories", "High Risk", "Medium Risk", "Low Risk"], index=0)
        with t_col2:
            contract_filter = st.selectbox("Filter Contract", ["All Contracts"] + list(scored_customers_df["Contract"].unique()), index=0)
        with t_col3:
            sort_order = st.selectbox("Sort by Churn Probability", ["Highest Risk First", "Lowest Risk First"])
        with t_col4:
            search_query = st.text_input("Search Customer ID", placeholder="e.g. 7590-VHVEG").strip()

        # Apply filters
        display_df = scored_customers_df.copy()
        if risk_filter != "All Categories":
            display_df = display_df[display_df["risk_category"] == risk_filter]
        if contract_filter != "All Contracts":
            display_df = display_df[display_df["Contract"] == contract_filter]
        if search_query:
            display_df = display_df[display_df["customerID"].str.contains(search_query, case=False, na=False)]

        ascending = True if sort_order == "Lowest Risk First" else False
        display_df = display_df.sort_values(by="churn_probability", ascending=ascending)

        render_cols = [
            "customerID", "tenure", "Contract", "InternetService", "PaymentMethod",
            "MonthlyCharges", "churn_probability", "risk_category"
        ]
        renamed_cols = {
            "customerID": "Customer ID",
            "tenure": "Tenure (Mos)",
            "Contract": "Contract",
            "InternetService": "Internet Service",
            "PaymentMethod": "Payment Method",
            "MonthlyCharges": "Monthly Bill ($)",
            "churn_probability": "Predicted Churn Probability",
            "risk_category": "Risk Category"
        }
        table_to_show = display_df[render_cols].rename(columns=renamed_cols)

        st.dataframe(
            table_to_show.style.format({
                "Monthly Bill ($)": "${:.2f}",
                "Predicted Churn Probability": "{:.3f}"
            }),
            width="stretch",
            height=400
        )

        st.caption(f"Displaying **{len(table_to_show):,}** scored customer records.")

        # CSV Download Button
        csv_bytes = table_to_show.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Export Filtered Portfolio Data (CSV)",
            data=csv_bytes,
            file_name="customer360_scored_portfolio.csv",
            mime="text/csv",
            help="Export the active view of scored accounts and prediction probabilities."
        )

    # ---------------- TAB 2: PREDICT NEW CUSTOMER ----------------
    with tab_predict_new:
        st.markdown("""
        <div style="margin-bottom: 12px;">
            <div style="font-size: 1.05rem; font-weight: 700; color: #0F172A;">Real-Time Account Scoring Engine</div>
            <div style="font-size: 0.82rem; color: #64748B;">Enter prospective or active customer characteristics to evaluate churn risk on demand via the FastAPI microservice.</div>
        </div>
        """, unsafe_allow_html=True)

        with st.form("new_customer_form", clear_on_submit=False):
            st.markdown("##### 1. Personal & Household Demographics")
            p_col1, p_col2, p_col3, p_col4 = st.columns(4)
            with p_col1:
                form_gender = st.selectbox("Gender", ["Female", "Male"], index=0)
            with p_col2:
                form_senior = st.selectbox("Senior Citizen", ["No", "Yes"], index=0)
            with p_col3:
                form_partner = st.selectbox("Partner", ["No", "Yes"], index=0)
            with p_col4:
                form_dependents = st.selectbox("Dependents", ["No", "Yes"], index=0)

            st.markdown("<hr style='margin: 12px 0; border: none; border-top: 1px solid #F1F5F9;'>", unsafe_allow_html=True)
            st.markdown("##### 2. Service Subscriptions & Add-Ons")
            s_col1, s_col2, s_col3 = st.columns(3)
            with s_col1:
                form_tenure = st.number_input(
                    "Tenure (Months with Company)",
                    min_value=0,
                    max_value=72,
                    value=12,
                    step=1,
                    help="Customer tenure between 0 and 72 months."
                )
                form_phone = st.selectbox("Phone Service", ["Yes", "No"], index=0)
                form_multilines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"], index=0)
                form_internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"], index=0)
            with s_col2:
                form_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"], index=0)
                form_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"], index=0)
                form_device = st.selectbox("Device Protection", ["No", "Yes", "No internet service"], index=0)
            with s_col3:
                form_tech = st.selectbox("Tech Support", ["No", "Yes", "No internet service"], index=0)
                form_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"], index=0)
                form_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"], index=0)

            st.markdown("<hr style='margin: 12px 0; border: none; border-top: 1px solid #F1F5F9;'>", unsafe_allow_html=True)
            st.markdown("##### 3. Account Terms & Billing Profile")
            a_col1, a_col2, a_col3, a_col4 = st.columns(4)
            with a_col1:
                form_contract = st.selectbox("Contract Terms", ["Month-to-month", "One year", "Two year"], index=0)
            with a_col2:
                form_paperless = st.selectbox("Paperless Billing", ["Yes", "No"], index=0)
            with a_col3:
                form_payment = st.selectbox(
                    "Payment Channel",
                    [
                        "Electronic check",
                        "Mailed check",
                        "Bank transfer (automatic)",
                        "Credit card (automatic)",
                    ],
                    index=0
                )
            with a_col4:
                form_monthly = st.number_input(
                    "Monthly Charges ($)",
                    min_value=0.0,
                    max_value=300.0,
                    value=70.0,
                    step=0.5,
                    format="%.2f"
                )

            # Auto-calculate suggested total charges
            suggested_total = round(float(form_tenure) * float(form_monthly), 2)
            t_col_a, t_col_b = st.columns([1, 3])
            with t_col_a:
                form_total = st.number_input(
                    "Total Charges ($)",
                    min_value=0.0,
                    max_value=20000.0,
                    value=suggested_total,
                    step=5.0,
                    format="%.2f"
                )
            with t_col_b:
                st.caption(
                    f"Suggested Historical Total: **${suggested_total:,.2f}** (tenure × monthly charges). "
                    "You may override this to match exact billing records."
                )

            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            submitted = st.form_submit_button("⚡ Calculate Churn Risk Score", width="stretch")

        if submitted:
            senior_val = 1 if form_senior == "Yes" else 0
            new_customer_payload = {
                "gender": form_gender,
                "SeniorCitizen": senior_val,
                "Partner": form_partner,
                "Dependents": form_dependents,
                "tenure": int(form_tenure),
                "PhoneService": form_phone,
                "MultipleLines": form_multilines,
                "InternetService": form_internet,
                "OnlineSecurity": form_security,
                "OnlineBackup": form_backup,
                "DeviceProtection": form_device,
                "TechSupport": form_tech,
                "StreamingTV": form_tv,
                "StreamingMovies": form_movies,
                "Contract": form_contract,
                "PaperlessBilling": form_paperless,
                "PaymentMethod": form_payment,
                "MonthlyCharges": float(form_monthly),
                "TotalCharges": float(form_total),
            }

            try:
                with st.spinner("Executing model inference via FastAPI microservice..."):
                    pred_result = predict_single_customer_via_api(new_customer_payload)

                prob = float(pred_result["churn_probability"])
                prob_pct = prob * 100
                churn_outcome = pred_result["predicted_churn"]
                risk_cat = pred_result["risk_category"]

                st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
                st.markdown("#### Real-Time Inference Results")

                # Outcome KPI summary row
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.markdown(f"""
                    <div class="kpi-card {'accent-red' if prob >= 0.70 else 'accent-amber' if prob >= 0.40 else 'accent-green'}">
                        <div class="kpi-label">Estimated Churn Probability</div>
                        <div class="kpi-value" style="color: {'#DC2626' if prob >= 0.70 else '#D97706' if prob >= 0.40 else '#16A34A'};">{prob_pct:.1f}%</div>
                        <div class="kpi-desc">Champion XGBoost Output</div>
                    </div>
                    """, unsafe_allow_html=True)
                with m2:
                    st.markdown(f"""
                    <div class="kpi-card accent-blue">
                        <div class="kpi-label">Predicted Outcome</div>
                        <div class="kpi-value" style="color: #0F172A;">{churn_outcome}</div>
                        <div class="kpi-desc">Classification decision</div>
                    </div>
                    """, unsafe_allow_html=True)
                with m3:
                    if risk_cat == "High Risk":
                        badge_html = '<span class="badge-high" style="font-size: 0.92rem; padding: 6px 14px;">High Risk (&ge;70%)</span>'
                    elif risk_cat == "Medium Risk":
                        badge_html = '<span class="badge-medium" style="font-size: 0.92rem; padding: 6px 14px;">Medium Risk (40-69%)</span>'
                    else:
                        badge_html = '<span class="badge-low" style="font-size: 0.92rem; padding: 6px 14px;">Low Risk (&lt;40%)</span>'

                    st.markdown(f"""
                    <div class="kpi-card {'accent-red' if risk_cat == 'High Risk' else 'accent-amber' if risk_cat == 'Medium Risk' else 'accent-green'}">
                        <div class="kpi-label">Risk Priority Tier</div>
                        <div style="margin: 8px 0;">{badge_html}</div>
                        <div class="kpi-desc">Action priority classification</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

                # Progress / Risk gauge representation
                st.progress(prob, text=f"Probability Exposure: {prob_pct:.1f}%")

                # Clear Tier Explanation
                if risk_cat == "High Risk":
                    exp_text = (
                        "<strong>High Risk Tier (≥ 70%):</strong> Account exhibits prominent attrition indicators. "
                        "Immediate retention outreach, contract lock incentive, or service optimization is recommended."
                    )
                elif risk_cat == "Medium Risk":
                    exp_text = (
                        "<strong>Medium Risk Tier (40% - 69%):</strong> Account displays moderate attrition potential. "
                        "Proactive relationship touchpoints and value reinforcement are suggested."
                    )
                else:
                    exp_text = (
                        "<strong>Low Risk Tier (< 40%):</strong> Account demonstrates favorable retention signals with low attrition risk. "
                        "Continue standard customer lifecycle engagement."
                    )

                st.markdown(f"""
                <div class="advisory-box" style="margin-top: 14px;">
                    <strong>Risk Category Assessment:</strong><br>{exp_text}<br><br>
                    <span style="font-size: 0.76rem; color: #64748B;"><em>System Note: {pred_result.get('message', 'Estimated model output.')}</em></span>
                </div>
                """, unsafe_allow_html=True)

                # Prescriptive Retention Recommendations
                st.markdown("##### Prescriptive Retention Recommendations")
                recs = []
                if form_contract == "Month-to-month":
                    recs.append("• **Contract Incentive**: Account is on a month-to-month agreement. Offer an annual contract with price-lock stability to anchor long-term tenure.")
                if form_payment == "Electronic check":
                    recs.append("• **Payment Channel Migration**: Account relies on electronic check. Guide enrollment into automated credit card or bank debit billing.")
                if int(form_tenure) <= 12:
                    recs.append("• **Early Onboarding Care**: Customer is within the high-attrition first year (≤12 mos). Proactively verify service satisfaction.")
                if form_internet == "Fiber optic" and float(form_monthly) >= 70.0:
                    recs.append("• **Fiber Service Quality Check**: Premium Fiber Optic subscriber with high billing. Ensure premium tech support coverage and connection stability.")
                if form_security == "No" or form_tech == "No":
                    recs.append("• **Value-Add Security Bundle**: Account lacks online security or technical support services. Provide trial bundles to increase ecosystem stickiness.")
                if not recs:
                    recs.append("• **Standard Account Care**: Profile demonstrates resilient attributes. Maintain proactive quarterly check-ins.")

                for r in recs:
                    st.markdown(r)

            except RuntimeError as api_err:
                st.error(f"**Prediction Service Error:** {api_err}")
            except Exception as exc:
                st.error(f"**Unexpected Error:** {exc}")


# =============================================================================
# 4. SECTION D: RETENTION INSIGHTS & STRATEGIC PLAYBOOKS
# =============================================================================
elif section == "Retention Insights":
    st.markdown("""
    <div class="dashboard-header">
        <div class="dashboard-title">Retention Insights & Strategic Playbooks</div>
        <div class="dashboard-subtitle">Actionable retention playbooks grounded in empirical attrition data and customer account diagnostics.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="advisory-box">
        <strong>Strategic Advisory:</strong>
        The playbooks below synthesize historical patterns and machine learning risk scoring into targeted retention interventions.
        These represent operational proposals designed for pilot rollout and A/B verification.
    </div>
    """, unsafe_allow_html=True)

    # Compute cohort counts dynamically
    c1_total = len(scored_customers_df[scored_customers_df["Contract"] == "Month-to-month"])
    c1_high = len(scored_customers_df[(scored_customers_df["Contract"] == "Month-to-month") & (scored_customers_df["risk_category"] == "High Risk")])

    c2_total = len(scored_customers_df[scored_customers_df["PaymentMethod"] == "Electronic check"])
    c2_high = len(scored_customers_df[(scored_customers_df["PaymentMethod"] == "Electronic check") & (scored_customers_df["risk_category"] == "High Risk")])

    c3_total = len(scored_customers_df[scored_customers_df["tenure"] <= 12])
    c3_high = len(scored_customers_df[(scored_customers_df["tenure"] <= 12) & (scored_customers_df["risk_category"] == "High Risk")])

    c4_total = len(scored_customers_df[scored_customers_df["InternetService"] == "Fiber optic"])
    c4_high = len(scored_customers_df[(scored_customers_df["InternetService"] == "Fiber optic") & (scored_customers_df["risk_category"] == "High Risk")])

    # 4 Structured Playbooks
    p1, p2 = st.columns(2)

    with p1:
        st.markdown(f"""
        <div class="playbook-card p-high">
            <div class="playbook-header">
                <div>
                    <div class="playbook-title">Playbook 1: Contract Migration & Term Lock</div>
                    <span class="badge-high" style="margin-top: 4px;">High Priority Segment</span>
                </div>
                <span class="cohort-badge">{c1_total:,} accounts ({c1_high:,} high risk)</span>
            </div>
            <div class="playbook-section-label">Target Customer Segment</div>
            <div class="playbook-body">Month-to-month contract subscribers, prioritizing accounts with predicted probability &ge; 0.70.</div>
            <div class="playbook-section-label">Empirical Churn Insight</div>
            <div class="playbook-body">Month-to-month subscribers exhibit <strong>42.71% historical churn</strong>, compared to 11.27% for one-year and 2.83% for two-year plans.</div>
            <div class="playbook-section-label">Prescribed Retention Actions</div>
            <div class="playbook-body">
                • Target accounts approaching month 3–6 with price-lock annual contract incentives.<br>
                • Structure tiered multi-year packages with introductory speed or streaming bundles.<br>
                • Position long-term agreements around guaranteed rate certainty.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with p2:
        st.markdown(f"""
        <div class="playbook-card p-med">
            <div class="playbook-header">
                <div>
                    <div class="playbook-title">Playbook 2: Automated Billing Conversion</div>
                    <span class="badge-medium" style="margin-top: 4px;">High Impact Friction</span>
                </div>
                <span class="cohort-badge">{c2_total:,} accounts ({c2_high:,} high risk)</span>
            </div>
            <div class="playbook-section-label">Target Customer Segment</div>
            <div class="playbook-body">Subscribers paying via electronic checks with elevated churn risk scores.</div>
            <div class="playbook-section-label">Empirical Churn Insight</div>
            <div class="playbook-body">Electronic check users exhibit <strong>45.29% historical churn</strong>, versus 15.24% for automated credit cards and 16.71% for automated bank debit.</div>
            <div class="playbook-section-label">Prescribed Retention Actions</div>
            <div class="playbook-body">
                • Promote seamless one-click autopay enrollment within the customer account portal.<br>
                • Provide a one-time bill credit incentive upon enrolling in automated debit.<br>
                • Replace manual payment reminder emails with automated billing confirmations.
            </div>
        </div>
        """, unsafe_allow_html=True)

    p3, p4 = st.columns(2)

    with p3:
        st.markdown(f"""
        <div class="playbook-card p-high">
            <div class="playbook-header">
                <div>
                    <div class="playbook-title">Playbook 3: Early Lifecycle Onboarding</div>
                    <span class="badge-high" style="margin-top: 4px;">Urgent Retention Window</span>
                </div>
                <span class="cohort-badge">{c3_total:,} accounts ({c3_high:,} high risk)</span>
            </div>
            <div class="playbook-section-label">Target Customer Segment</div>
            <div class="playbook-body">New subscribers with tenure between 0 and 12 months.</div>
            <div class="playbook-section-label">Empirical Churn Insight</div>
            <div class="playbook-body">Departing customers have a median tenure of only 10.0 months (vs 38.0 months for retained). Churn peaks at 51.9% during month 0–1.</div>
            <div class="playbook-section-label">Prescribed Retention Actions</div>
            <div class="playbook-body">
                • Deploy a 30-day proactive onboarding health check after installation.<br>
                • Route first-year support tickets to senior technical specialists.<br>
                • Trigger automated network speed satisfaction surveys at days 15 and 45.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with p4:
        st.markdown(f"""
        <div class="playbook-card p-info">
            <div class="playbook-header">
                <div>
                    <div class="playbook-title">Playbook 4: Premium Fiber Quality Assurance</div>
                    <span class="badge-low" style="background:#EFF6FF;color:#1D4ED8;border-color:#BFDBFE;margin-top:4px;">Revenue Defense</span>
                </div>
                <span class="cohort-badge">{c4_total:,} accounts ({c4_high:,} high risk)</span>
            </div>
            <div class="playbook-section-label">Target Customer Segment</div>
            <div class="playbook-body">Fiber Optic internet subscribers paying premium rates (MonthlyCharges &ge; $70).</div>
            <div class="playbook-section-label">Empirical Churn Insight</div>
            <div class="playbook-body">Fiber Optic subscribers exhibit <strong>41.89% historical churn</strong>, despite fiber representing the highest tier offering.</div>
            <div class="playbook-section-label">Prescribed Retention Actions</div>
            <div class="playbook-body">
                • Audit line stability and Wi-Fi router performance for high-spend accounts.<br>
                • Bundle free Online Security and Tech Support to reinforce perceived value.<br>
                • Establish proactive outreach whenever local line maintenance occurs.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # Individual Customer Diagnostics
    st.markdown("#### Individual Account Retention Inspector")
    st.caption("Inspect individual customer profiles to review predicted risk, tenure history, and contextual intervention proposals.")

    sample_ids = scored_customers_df["customerID"].tolist()
    default_index = 2 if len(sample_ids) > 2 else 0
    selected_cid = st.selectbox("Select or Search Customer ID", sample_ids, index=default_index)

    cust_row = scored_customers_df[scored_customers_df["customerID"] == selected_cid].iloc[0]

    # Customer Attributes Row
    ic1, ic2, ic3, ic4, ic5 = st.columns(5)
    with ic1:
        st.markdown(f"""
        <div class="kpi-card accent-blue">
            <div class="kpi-label">Customer ID</div>
            <div class="kpi-value" style="font-size: 1.15rem;">{cust_row['customerID']}</div>
            <div class="kpi-desc">Account Identifier</div>
        </div>
        """, unsafe_allow_html=True)
    with ic2:
        st.markdown(f"""
        <div class="kpi-card accent-blue">
            <div class="kpi-label">Tenure</div>
            <div class="kpi-value" style="font-size: 1.45rem;">{cust_row['tenure']} <span style="font-size: 0.9rem; font-weight: 500;">Mos</span></div>
            <div class="kpi-desc">Account Age</div>
        </div>
        """, unsafe_allow_html=True)
    with ic3:
        st.markdown(f"""
        <div class="kpi-card accent-blue">
            <div class="kpi-label">Monthly Bill</div>
            <div class="kpi-value" style="font-size: 1.45rem;">${cust_row['MonthlyCharges']:.2f}</div>
            <div class="kpi-desc">Current Recurring Plan</div>
        </div>
        """, unsafe_allow_html=True)
    with ic4:
        prob_val = cust_row['churn_probability'] * 100
        st.markdown(f"""
        <div class="kpi-card {'accent-red' if prob_val >= 70 else 'accent-amber' if prob_val >= 40 else 'accent-green'}">
            <div class="kpi-label">Estimated Churn Risk</div>
            <div class="kpi-value" style="font-size: 1.45rem; color: {'#DC2626' if prob_val >= 70 else '#D97706' if prob_val >= 40 else '#16A34A'};">{prob_val:.1f}%</div>
            <div class="kpi-desc">XGBoost Probability</div>
        </div>
        """, unsafe_allow_html=True)
    with ic5:
        category = cust_row["risk_category"]
        if category == "High Risk":
            b_html = '<span class="badge-high" style="font-size:0.88rem;padding:5px 12px;">High Risk</span>'
        elif category == "Medium Risk":
            b_html = '<span class="badge-medium" style="font-size:0.88rem;padding:5px 12px;">Medium Risk</span>'
        else:
            b_html = '<span class="badge-low" style="font-size:0.88rem;padding:5px 12px;">Low Risk</span>'

        st.markdown(f"""
        <div class="kpi-card {'accent-red' if category == 'High Risk' else 'accent-amber' if category == 'Medium Risk' else 'accent-green'}">
            <div class="kpi-label">Priority Tier</div>
            <div style="margin: 6px 0;">{b_html}</div>
            <div class="kpi-desc">Classification</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Contextual Action Proposals for this Customer
    st.markdown(f"**Proposed Retention Focus for Account `{selected_cid}`:**")
    recs = []
    if cust_row["Contract"] == "Month-to-month":
        recs.append("• **Contract Migration**: Customer is on a month-to-month plan. Propose an annual plan with rate stability guarantees.")
    if cust_row["PaymentMethod"] == "Electronic check":
        recs.append("• **Autopay Transition**: Customer pays via electronic check. Encourage automated credit card or bank debit billing.")
    if cust_row["tenure"] <= 12:
        recs.append("• **Early Lifecycle Check**: Customer is within the high-risk first year (≤12 months). Conduct proactive satisfaction check-in.")
    if cust_row["InternetService"] == "Fiber optic" and cust_row["MonthlyCharges"] >= 70.0:
        recs.append("• **Fiber Service Experience**: Customer is a high-spend Fiber Optic subscriber. Ensure service performance meets quality standards.")
    if not recs:
        recs.append("• **Stable Account Baseline**: Customer displays low estimated churn risk and stable tenure. Continue standard lifecycle engagement.")

    for r in recs:
        st.markdown(r)


# =============================================================================
# 5. FOOTER & MODEL GOVERNANCE (COMPACT ENTERPRISE FOOTER)
# =============================================================================
st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
st.markdown("---")

f_col1, f_col2, f_col3 = st.columns(3)

with f_col1:
    st.markdown("""
    <div class="info-card">
        <div class="info-card-title">Model Specifications</div>
        <strong>Architecture:</strong> Champion XGBoost Classifier<br>
        <strong>Inference Engine:</strong> FastAPI REST Microservice<br>
        <strong>Dataset:</strong> IBM Telco Customer Churn (7,043 Records)<br>
        <strong>Target Variable:</strong> Customer Churn (Yes / No)
    </div>
    """, unsafe_allow_html=True)

with f_col2:
    st.markdown("""
    <div class="info-card">
        <div class="info-card-title">Held-Out Test Set Metrics</div>
        <strong>Test Accuracy:</strong> 75.88%<br>
        <strong>Test Precision:</strong> 52.94%<br>
        <strong>Test Recall:</strong> 80.36%<br>
        <strong>Test F1-Score:</strong> 63.83%<br>
        <strong>Test ROC-AUC:</strong> 85.25% &bull; <strong>PR-AUC:</strong> 66.51%
    </div>
    """, unsafe_allow_html=True)

with f_col3:
    st.markdown("""
    <div class="info-card">
        <div class="info-card-title">Operational Risk Thresholds</div>
        <strong>High Risk Tier:</strong> Estimated Probability &ge; 0.70<br>
        <strong>Medium Risk Tier:</strong> 0.40 &le; Probability &lt; 0.70<br>
        <strong>Low Risk Tier:</strong> Estimated Probability &lt; 0.40
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="advisory-box" style="margin-top: 12px; font-size: 0.78rem;">
    <strong>Governance Note:</strong> Model probabilities and risk tiers are analytical estimations designed for operational testing, not absolute determinants. Test metrics are derived from the held-out evaluation baseline.
</div>
""", unsafe_allow_html=True)
