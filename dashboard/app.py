import json
from pathlib import Path

import pandas as pd
import requests
import streamlit as st


st.set_page_config(
    page_title="Fraud Risk Prioritization System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "http://127.0.0.1:8000/predict"
RANKING_FILE = Path("outputs/reports/ranking_metrics.json")
LGBM_METRICS_FILE = Path("outputs/reports/lgbm_metrics.json")
SHAP_FILE = Path("outputs/reports/shap_feature_importance.csv")
SHAP_PLOT_FILE = Path("outputs/plots/shap_summary.png")


def load_json_file(path: Path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def risk_badge_html(risk_level: str) -> str:
    if risk_level == "high":
        return '<div class="risk-badge risk-high">HIGH RISK</div>'
    if risk_level == "medium":
        return '<div class="risk-badge risk-medium">MEDIUM RISK</div>'
    return '<div class="risk-badge risk-low">LOW RISK</div>'


def alert_text(prob: float) -> str:
    if prob >= 0.8:
        return "High-confidence suspicious transaction. Immediate analyst review is recommended."
    if prob >= 0.4:
        return "Elevated risk detected. Prioritize this transaction for queue-based review."
    return "Current model evidence suggests this is a low-priority transaction."


# -----------------------------
# Load project metrics
# -----------------------------
lgbm_metrics = load_json_file(LGBM_METRICS_FILE)
ranking_metrics = load_json_file(RANKING_FILE)

roc_auc = lgbm_metrics.get("roc_auc") if lgbm_metrics else None
pr_auc = lgbm_metrics.get("pr_auc") if lgbm_metrics else None
n_features = lgbm_metrics.get("n_features") if lgbm_metrics else None
precision_100 = ranking_metrics.get("precision@100") if ranking_metrics else None

roc_auc_text = f"{roc_auc:.4f}" if roc_auc is not None else "N/A"
pr_auc_text = f"{pr_auc:.4f}" if pr_auc is not None else "N/A"
n_features_text = str(n_features) if n_features is not None else "N/A"
precision_100_text = f"{precision_100:.4f}" if precision_100 is not None else "N/A"

# queue insights
top50_precision = ranking_metrics.get("precision@50") if ranking_metrics else None
top100_precision = ranking_metrics.get("precision@100") if ranking_metrics else None
top200_precision = ranking_metrics.get("precision@200") if ranking_metrics else None

top50_precision_text = f"{top50_precision:.2%}" if top50_precision is not None else "N/A"
top100_precision_text = f"{top100_precision:.2%}" if top100_precision is not None else "N/A"
top200_precision_text = f"{top200_precision:.2%}" if top200_precision is not None else "N/A"


# -----------------------------
# Premium CSS theme
# -----------------------------
st.markdown(
    """
<style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 1.6rem;
        max-width: 1380px;
    }

    .main {
        background:
            radial-gradient(circle at top left, rgba(30,64,175,0.10), transparent 25%),
            radial-gradient(circle at top right, rgba(220,38,38,0.08), transparent 20%),
            linear-gradient(180deg, #040814 0%, #060b17 100%);
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1220 0%, #111827 100%);
        border-right: 1px solid rgba(148,163,184,0.08);
    }

    .sidebar-header {
        color: #f8fafc;
        font-size: 1.12rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }

    .sidebar-subtext {
        color: #94a3b8;
        font-size: 0.92rem;
        line-height: 1.45;
        margin-bottom: 0.8rem;
    }

    .hero {
        background:
            linear-gradient(135deg, rgba(10,15,28,0.97), rgba(17,24,39,0.95)),
            radial-gradient(circle at right top, rgba(59,130,246,0.20), transparent 35%);
        border: 1px solid rgba(96,165,250,0.12);
        border-radius: 22px;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 12px 30px rgba(2,8,23,0.35);
        margin-bottom: 1rem;
    }

    .hero-tag {
        display: inline-block;
        background: rgba(59,130,246,0.14);
        color: #93c5fd;
        border: 1px solid rgba(59,130,246,0.18);
        border-radius: 999px;
        padding: 0.28rem 0.7rem;
        font-size: 0.78rem;
        font-weight: 700;
        margin-bottom: 0.7rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    .hero-title {
        color: #f8fafc;
        font-size: 2.1rem;
        font-weight: 800;
        margin-bottom: 0.4rem;
    }

    .hero-subtitle {
        color: #cbd5e1;
        font-size: 1rem;
        line-height: 1.55;
        max-width: 900px;
    }

    .status-strip {
        background: rgba(8,15,28,0.85);
        border: 1px solid rgba(148,163,184,0.08);
        border-radius: 16px;
        padding: 0.85rem 1rem;
        margin-bottom: 1rem;
    }

    .status-label {
        color: #94a3b8;
        font-size: 0.84rem;
        margin-bottom: 0.18rem;
    }

    .status-value {
        color: #e2e8f0;
        font-size: 0.98rem;
        font-weight: 700;
    }

    .live-pill {
        display: inline-block;
        background: rgba(34,197,94,0.14);
        color: #4ade80;
        border: 1px solid rgba(34,197,94,0.20);
        border-radius: 999px;
        padding: 0.36rem 0.82rem;
        font-size: 0.88rem;
        font-weight: 800;
        text-align: center;
        min-width: 88px;
    }

    .kpi-card {
        background: linear-gradient(180deg, #091428 0%, #0b1220 100%);
        border: 1px solid rgba(59,130,246,0.10);
        border-radius: 18px;
        padding: 1rem;
        text-align: center;
        min-height: 112px;
        box-shadow: 0 10px 24px rgba(2,8,23,0.30);
    }

    .kpi-label {
        color: #93c5fd;
        font-size: 0.9rem;
        margin-bottom: 0.45rem;
    }

    .kpi-value {
        color: #f8fafc;
        font-size: 1.7rem;
        font-weight: 800;
    }

    .card {
        background: linear-gradient(180deg, #08101e 0%, #0b1220 100%);
        border: 1px solid rgba(148,163,184,0.08);
        border-radius: 20px;
        padding: 1.15rem 1.15rem 1rem 1.15rem;
        margin-bottom: 1rem;
        box-shadow: 0 10px 26px rgba(2,8,23,0.28);
    }

    .card-note {
        color: #94a3b8;
        font-size: 0.92rem;
        line-height: 1.5;
        margin-bottom: 0.85rem;
    }

    .score-label {
        color: #93c5fd;
        font-size: 0.95rem;
        margin-bottom: 0.28rem;
    }

    .score-value {
        color: #f8fafc;
        font-size: 2.35rem;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 0.2rem;
    }

    .risk-badge {
        display: inline-block;
        padding: 0.5rem 0.95rem;
        border-radius: 999px;
        font-size: 0.94rem;
        font-weight: 800;
        border: 1px solid transparent;
    }

    .risk-high {
        background: rgba(239,68,68,0.14);
        color: #f87171;
        border-color: rgba(239,68,68,0.20);
    }

    .risk-medium {
        background: rgba(245,158,11,0.14);
        color: #fbbf24;
        border-color: rgba(245,158,11,0.20);
    }

    .risk-low {
        background: rgba(34,197,94,0.14);
        color: #4ade80;
        border-color: rgba(34,197,94,0.20);
    }

    .insight-box {
        background: rgba(10,15,28,0.78);
        border: 1px solid rgba(59,130,246,0.12);
        border-radius: 14px;
        padding: 0.95rem 1rem;
        margin-top: 0.9rem;
    }

    .insight-title {
        color: #93c5fd;
        font-size: 0.82rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.3rem;
    }

    .insight-text {
        color: #e2e8f0;
        font-size: 0.96rem;
        line-height: 1.5;
    }

    .mini-stat {
        background: rgba(10,15,28,0.82);
        border: 1px solid rgba(148,163,184,0.08);
        border-radius: 14px;
        padding: 0.8rem 0.9rem;
        text-align: center;
    }

    .mini-stat-label {
        color: #94a3b8;
        font-size: 0.84rem;
        margin-bottom: 0.2rem;
    }

    .mini-stat-value {
        color: #f8fafc;
        font-size: 1.15rem;
        font-weight: 800;
    }

    .stButton > button {
        width: 100%;
        border-radius: 14px;
        padding: 0.8rem 1rem;
        font-weight: 800;
        border: 1px solid rgba(59,130,246,0.20);
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        color: white;
        box-shadow: 0 8px 20px rgba(37,99,235,0.24);
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #1d4ed8, #1e40af);
        color: white;
    }
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.markdown('<div class="sidebar-header">⚙️ Investigation Input Panel</div>', unsafe_allow_html=True)
st.sidebar.markdown(
    '<div class="sidebar-subtext">Configure transaction attributes and send them to the live fraud scoring service.</div>',
    unsafe_allow_html=True,
)

transaction_id = st.sidebar.number_input("Transaction ID", value=999001, step=1)
transaction_dt = st.sidebar.number_input("TransactionDT", value=86400.0)
transaction_amt = st.sidebar.number_input("Transaction Amount", value=2500.0)

st.sidebar.markdown("---")
st.sidebar.subheader("Card Signals")
card1 = st.sidebar.number_input("card1", value=15000.0)
card2 = st.sidebar.number_input("card2", value=111.0)
card3 = st.sidebar.number_input("card3", value=150.0)
card5 = st.sidebar.number_input("card5", value=226.0)

st.sidebar.markdown("---")
st.sidebar.subheader("Location / Identity Signals")
addr1 = st.sidebar.number_input("addr1", value=315.0)
dist1 = st.sidebar.number_input("dist1", value=19.0)
p_emaildomain = st.sidebar.text_input("P_emaildomain", value="gmail.com")

payload = {
    "TransactionID": int(transaction_id),
    "TransactionDT": float(transaction_dt),
    "TransactionAmt": float(transaction_amt),
    "card1": float(card1),
    "card2": float(card2),
    "card3": float(card3),
    "card5": float(card5),
    "addr1": float(addr1),
    "dist1": float(dist1),
    "P_emaildomain": p_emaildomain,
}


# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
<div class="hero">
    <div class="hero-tag">Fraud Monitoring Console</div>
    <div class="hero-title">🛡️ Fraud Risk Prioritization System</div>
    <div class="hero-subtitle">
        Security-oriented ML dashboard for transaction fraud scoring, investigation queue prioritization,
        model explainability, and operational fraud analysis.
    </div>
</div>
""",
    unsafe_allow_html=True,
)

s1, s2 = st.columns([0.85, 0.15])

with s1:
    st.markdown(
        """
<div class="status-strip">
    <div class="status-label">System Status</div>
    <div class="status-value">Model artifacts loaded • API expected at 127.0.0.1:8000 • Dashboard ready for live scoring</div>
</div>
""",
        unsafe_allow_html=True,
    )

with s2:
    st.markdown(
        """
<div style="display:flex;justify-content:center;align-items:center;height:100%;">
    <div class="live-pill">● LIVE</div>
</div>
""",
        unsafe_allow_html=True,
    )


# -----------------------------
# KPI row
# -----------------------------
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(
        f"""
<div class="kpi-card">
    <div class="kpi-label">Validation ROC-AUC</div>
    <div class="kpi-value">{roc_auc_text}</div>
</div>
""",
        unsafe_allow_html=True,
    )

with k2:
    st.markdown(
        f"""
<div class="kpi-card">
    <div class="kpi-label">Validation PR-AUC</div>
    <div class="kpi-value">{pr_auc_text}</div>
</div>
""",
        unsafe_allow_html=True,
    )

with k3:
    st.markdown(
        f"""
<div class="kpi-card">
    <div class="kpi-label">Features Used</div>
    <div class="kpi-value">{n_features_text}</div>
</div>
""",
        unsafe_allow_html=True,
    )

with k4:
    st.markdown(
        f"""
<div class="kpi-card">
    <div class="kpi-label">Precision@100</div>
    <div class="kpi-value">{precision_100_text}</div>
</div>
""",
        unsafe_allow_html=True,
    )


# -----------------------------
# Main layout
# -----------------------------
left, right = st.columns([1.12, 0.88])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🔎 Live Transaction Risk Assessment")
    st.markdown(
        '<div class="card-note">Send a transaction to the FastAPI inference service and inspect the predicted fraud probability and investigation guidance.</div>',
        unsafe_allow_html=True,
    )

    result = None

    if st.button("Run Fraud Risk Assessment"):
        try:
            response = requests.post(API_URL, json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()
            st.session_state["latest_result"] = result
        except requests.exceptions.RequestException as e:
            st.error(f"API Error: {e}")
        except Exception as e:
            st.error(f"Unexpected Error: {e}")

    if "latest_result" in st.session_state:
        result = st.session_state["latest_result"]

    if result:
        fraud_probability = result["fraud_probability"]
        risk_level = result["risk_level"]

        p1, p2 = st.columns([1, 1])

        with p1:
            st.markdown('<div class="score-label">Fraud Probability</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="score-value">{fraud_probability:.4f}</div>',
                unsafe_allow_html=True,
            )

        with p2:
            st.markdown("**Predicted Risk Level**")
            st.markdown(risk_badge_html(risk_level), unsafe_allow_html=True)

        st.markdown(
            f"""
<div class="insight-box">
    <div class="insight-title">Investigation Guidance</div>
    <div class="insight-text">{alert_text(fraud_probability)}</div>
</div>
""",
            unsafe_allow_html=True,
        )

        rp1, rp2 = st.columns(2)

        with rp1:
            st.markdown("### Request Payload")
            st.json(payload)

        with rp2:
            st.markdown("### API Response")
            st.json(result)

    else:
        st.info("Run a fraud risk assessment to display the latest prediction.")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Explainability Console")
    st.markdown(
        '<div class="card-note">Visual explanation of how the model weighs transaction-level fraud signals using SHAP-based attribution.</div>',
        unsafe_allow_html=True,
    )

    if SHAP_PLOT_FILE.exists():
        st.image(str(SHAP_PLOT_FILE), use_container_width=True)
    else:
        st.info("SHAP summary plot not found yet.")

    if SHAP_FILE.exists():
        shap_df = pd.read_csv(SHAP_FILE)
        st.markdown("### Top SHAP Drivers")
        st.dataframe(shap_df.head(10), use_container_width=True, height=250)
    else:
        st.info("SHAP feature importance file not found yet.")

    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🏆 Investigation Queue Quality")
    st.markdown(
        '<div class="card-note">Operational ranking performance showing how effectively the model pushes suspicious transactions to the top of the analyst review queue.</div>',
        unsafe_allow_html=True,
    )

    qs1, qs2, qs3 = st.columns(3)

    with qs1:
        st.markdown(
            f"""
<div class="mini-stat">
    <div class="mini-stat-label">Top 50 Precision</div>
    <div class="mini-stat-value">{top50_precision_text}</div>
</div>
""",
            unsafe_allow_html=True,
        )

    with qs2:
        st.markdown(
            f"""
<div class="mini-stat">
    <div class="mini-stat-label">Top 100 Precision</div>
    <div class="mini-stat-value">{top100_precision_text}</div>
</div>
""",
            unsafe_allow_html=True,
        )

    with qs3:
        st.markdown(
            f"""
<div class="mini-stat">
    <div class="mini-stat-label">Top 200 Precision</div>
    <div class="mini-stat-value">{top200_precision_text}</div>
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    if ranking_metrics:
        rank_df = pd.DataFrame(
            {
                "Metric": list(ranking_metrics.keys()),
                "Value": list(ranking_metrics.values()),
            }
        )
        rank_df["Value"] = rank_df["Value"].map(lambda x: round(x, 4))
        st.dataframe(rank_df, use_container_width=True, height=300)
    else:
        st.info("Ranking metrics file not found yet.")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🚨 Analyst Interpretation")
    st.markdown(
        """
<div class="card-note">
This dashboard is designed like a lightweight fraud analyst console:
high-risk cases are scored in real time, queue quality is visible immediately,
and SHAP explains the strongest behavioral signals behind the model’s decisions.
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)