"""
AgriField PatchTST — Professional Dashboard
Design: GrainGuard AI enterprise design system (sample/crop_harvest)
Run:    streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agri_system.config import FEATURE_COLS, TARGET_CROPS

OUTPUTS_DIR   = ROOT / "artifacts" / "outputs"
PROCESSED_DIR = ROOT / "processed"

# ── page config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AgriField · PatchTST",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens (GrainGuard AI design system) ────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ── CSS custom properties (GrainGuard light theme) ── */
:root {
  --bg-app:              #f8fafc;
  --bg-surface:          #ffffff;
  --bg-surface-elevated: #f1f5f9;
  --bg-surface-subtle:   #f8fafc;
  --border-color:        #e2e8f0;
  --border-subtle:       #cbd5e1;
  --text-primary:        #0f172a;
  --text-secondary:      #475569;
  --text-muted:          #64748b;
  --accent-green:        #15803d;
  --accent-green-subtle: #f0fdf4;
  --accent-green-border: #bbf7d0;
  --card-shadow:         0 1px 3px 0 rgba(0,0,0,.05),0 1px 2px -1px rgba(0,0,0,.05);
}

/* ── Global resets ── */
html,body,[data-testid="stAppViewContainer"],[data-testid="stMain"],.main,.block-container {
  background: var(--bg-app) !important;
  color: var(--text-primary) !important;
  font-family: Inter, ui-sans-serif, system-ui, sans-serif !important;
}
/* Force all text in main area to be visible */
[data-testid="stMain"] p,
[data-testid="stMain"] span,
[data-testid="stMain"] label,
[data-testid="stMain"] div,
[data-testid="stMain"] li {
  color: var(--text-primary);
}
.block-container { padding-top: 0 !important; max-width: 1280px !important; }

/* ── Top header ── */
[data-testid="stHeader"] {
  background: var(--bg-surface) !important;
  border-bottom: 1px solid var(--border-color) !important;
}
[data-testid="stHeader"]::before {
  content: "AgriField  ·  PatchTST Multivariate Environmental Prediction";
  display: block; text-align: center; width: 100%;
  padding: 0 1rem; font-size: .85rem; font-weight: 700;
  color: var(--text-primary); line-height: 3.5rem; letter-spacing: -.01em;
}
[data-testid="stDecoration"] { display: none !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: var(--bg-surface) !important;
  border-right: 1px solid var(--border-color) !important;
}
[data-testid="stSidebar"] * {
  color: var(--text-primary) !important;
}
[data-testid="stSidebar"] label {
  color: var(--text-secondary) !important;
  font-size: .75rem !important; font-weight: 600 !important;
  text-transform: uppercase; letter-spacing: .04em;
}
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stSelectbox > div > div > div {
  background: var(--bg-surface-elevated) !important;
  border: 1px solid var(--border-color) !important;
  border-radius: 6px !important;
  color: var(--text-primary) !important;
  font-size: .82rem !important;
}
[data-testid="stSidebar"] .stSelectbox span,
[data-testid="stSidebar"] .stSelectbox p {
  color: var(--text-primary) !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label,
[data-testid="stSidebar"] [data-testid="stRadio"] p,
[data-testid="stSidebar"] [data-testid="stRadio"] span {
  color: var(--text-primary) !important;
  font-size: .82rem !important;
}
[data-testid="stSidebar"] [data-testid="stSlider"] p,
[data-testid="stSidebar"] [data-testid="stSlider"] span {
  color: var(--text-secondary) !important;
}
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] .stCaption {
  color: var(--text-muted) !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
  background: var(--bg-surface) !important;
  border: 1px solid var(--border-color) !important;
  border-radius: 8px !important;
  padding: 14px 16px !important;
  box-shadow: var(--card-shadow) !important;
}
[data-testid="stMetricLabel"] {
  font-size: .7rem !important; font-weight: 600 !important;
  text-transform: uppercase; letter-spacing: .05em;
  color: var(--text-muted) !important;
}
[data-testid="stMetricValue"] {
  font-size: 1.5rem !important; font-weight: 700 !important;
  color: var(--text-primary) !important; letter-spacing: -.02em;
}
[data-testid="stMetricDelta"] { font-size: .7rem !important; }

/* ── Tabs ── */
[data-testid="stTabs"] button {
  font-size: .78rem !important; font-weight: 500 !important;
  color: var(--text-muted) !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
  color: var(--accent-green) !important;
  border-bottom: 2px solid var(--accent-green) !important;
  font-weight: 700 !important;
}

/* ── Primary button ── */
[data-testid="baseButton-primary"] {
  background: #16a34a !important; color: #fff !important;
  border: none !important; border-radius: 6px !important;
  font-size: .8rem !important; font-weight: 600 !important;
  letter-spacing: .02em;
}
[data-testid="baseButton-primary"]:hover { background: #15803d !important; }

/* ── Secondary button ── */
[data-testid="baseButton-secondary"] {
  background: var(--bg-surface-elevated) !important;
  border: 1px solid var(--border-color) !important;
  color: var(--text-secondary) !important;
  border-radius: 6px !important; font-size: .78rem !important;
}

/* ── Dividers ── */
hr { border-color: var(--border-color) !important; margin: .75rem 0 !important; }

/* ── Headings ── */
h1,h2,h3,h4 {
  color: var(--text-primary) !important;
  letter-spacing: -.015em; font-weight: 600;
}

/* ── Expander ── */
[data-testid="stExpander"] {
  border: 1px solid var(--border-color) !important;
  border-radius: 8px !important;
  background: var(--bg-surface) !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
  border: 1px solid var(--border-color) !important;
  border-radius: 8px !important; overflow: hidden !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
  border: 1px solid var(--border-color) !important;
  border-radius: 8px !important;
  background: var(--bg-surface) !important;
}
[data-testid="stChatMessage"] {
  background: var(--bg-surface) !important;
  border: 1px solid var(--border-color) !important;
  border-radius: 8px !important;
}

/* ── Progress bar ── */
[data-testid="stProgressBar"] > div { background: #16a34a !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-surface-subtle); }
::-webkit-scrollbar-thumb { background: var(--border-subtle); border-radius: 4px; }

/* ── Slider ── */
[data-testid="stSlider"] .st-bd { background: #16a34a !important; }
</style>
""", unsafe_allow_html=True)

# ── Feature metadata ───────────────────────────────────────────────────
FEAT_META = {
    "temperature":   {"label": "Temperature",   "unit": "°C",    "color": "#ef4444"},
    "humidity":      {"label": "Humidity",       "unit": "%",     "color": "#3b82f6"},
    "soil_moisture": {"label": "Soil Moisture",  "unit": "%",     "color": "#16a34a"},
    "ph":            {"label": "Soil pH",        "unit": "",      "color": "#8b5cf6"},
    "N":             {"label": "Nitrogen (N)",   "unit": "mg/kg", "color": "#f59e0b"},
    "P":             {"label": "Phosphorus (P)", "unit": "mg/kg", "color": "#d97706"},
    "K":             {"label": "Potassium (K)",  "unit": "mg/kg", "color": "#7c3aed"},
    "rainfall":      {"label": "Rainfall",       "unit": "mm",    "color": "#0ea5e9"},
}

# ── Model comparison (paper Fig.3 & Fig.5) ────────────────────────────
MODELS    = ["CNN", "GRU", "LSTM", "Bi-LSTM", "PatchTST"]
RMSE_MULT = {"CNN": 2.20, "GRU": 1.72, "LSTM": 1.45, "Bi-LSTM": 1.20, "PatchTST": 1.00}
MAE_MULT  = {"CNN": 2.10, "GRU": 1.65, "LSTM": 1.40, "Bi-LSTM": 1.18, "PatchTST": 1.00}
R2_PAPER  = {"CNN": 0.441, "GRU": 0.663, "LSTM": 0.827, "Bi-LSTM": 0.938, "PatchTST": 0.978}

# ── Plotly white layout ────────────────────────────────────────────────
PL = dict(
    paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
    font=dict(family="Inter, sans-serif", color="#475569", size=11),
    xaxis=dict(gridcolor="#f1f5f9", linecolor="#e2e8f0", tickfont=dict(color="#64748b")),
    yaxis=dict(gridcolor="#f1f5f9", linecolor="#e2e8f0", tickfont=dict(color="#64748b")),
    margin=dict(l=8, r=8, t=32, b=8),
    legend=dict(bgcolor="#ffffff", bordercolor="#e2e8f0", font=dict(color="#475569", size=11)),
)


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

def _rgba(hex_c: str, a: float) -> str:
    h = hex_c.lstrip("#")
    r,g,b = int(h[0:2],16),int(h[2:4],16),int(h[4:6],16)
    return f"rgba({r},{g},{b},{a})"

def card(content: str, accent: str = "#e2e8f0") -> str:
    return (f"<div style='background:#fff;border:1px solid #e2e8f0;"
            f"border-left:3px solid {accent};border-radius:8px;"
            f"padding:14px 16px;box-shadow:0 1px 3px rgba(0,0,0,.04)'>"
            f"{content}</div>")

def badge(text: str, color: str, bg: str, border: str) -> str:
    return (f"<span style='background:{bg};color:{color};border:1px solid {border};"
            f"border-radius:4px;padding:2px 8px;font-size:.7rem;font-weight:700;"
            f"text-transform:uppercase;letter-spacing:.04em'>{text}</span>")


# ══════════════════════════════════════════════════════════════════════
# Data helpers
# ══════════════════════════════════════════════════════════════════════

@st.cache_data
def get_fields(crop: str) -> list[str]:
    p = PROCESSED_DIR / crop.lower() / "metadata.json"
    return json.loads(p.read_text())["fields"]

@st.cache_data
def get_metrics(crop: str) -> dict:
    p = OUTPUTS_DIR / crop.lower() / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else {}

@st.cache_data
def get_history(crop: str, field_id: str, n: int = 30) -> pd.DataFrame:
    df = pd.read_csv(
        PROCESSED_DIR / crop.lower() / f"{crop.lower()}_processed.csv",
        parse_dates=["date"],
    )
    return df[df["field_id"] == field_id].sort_values("date").tail(n).reset_index(drop=True)

@st.cache_data
def get_predictions(crop: str) -> dict | None:
    p = OUTPUTS_DIR / crop.lower() / "predictions.npz"
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    return {k: d[k] for k in d.files}

def run_forecast(crop: str, field_id: str) -> dict:
    """
    Cloud-safe forecast: reads from pre-saved predictions.npz and processed CSV.
    No torch required — all inference was done during training and saved to disk.
    """
    import json as _json

    # ── Load processed field data ─────────────────────────────────────
    df = pd.read_csv(
        PROCESSED_DIR / crop.lower() / f"{crop.lower()}_processed.csv",
        parse_dates=["date"],
    )
    field_df = df[df["field_id"] == field_id].sort_values("date").reset_index(drop=True)
    if field_df.empty:
        raise RuntimeError(f"Field {field_id} not found for crop {crop}")

    latest = field_df.iloc[-1]
    cur_conditions = {col: float(latest[col]) for col in FEATURE_COLS}

    # ── Load pre-saved predictions ────────────────────────────────────
    pred_data = get_predictions(crop)
    if pred_data is None:
        raise RuntimeError("predictions.npz not found — run evaluate.py first")

    feat_cols = list(pred_data["feature_cols"])
    field_ids = list(pred_data["field_ids"])

    # Find prediction windows for this field
    matching = [i for i, fid in enumerate(field_ids) if fid == field_id]
    if not matching:
        matching = list(range(min(7, len(field_ids))))  # fallback: first windows

    # Use the last matching window (most recent)
    win_idx = matching[-1]
    pred_7  = pred_data["pred"][win_idx]    # shape (7, 8)
    start   = str(pred_data["pred_starts"][win_idx])

    # Build forecast rows (7 days)
    import pandas as _pd
    forecast_dates = pd.date_range(start=start, periods=7, freq="D")
    forecast_rows  = []
    for i in range(7):
        row = {"date": str(forecast_dates[i].date())}
        for j, feat in enumerate(feat_cols):
            row[feat] = float(pred_7[i, j])
        forecast_rows.append(row)

    # ── Suitability assessment (rule-based, no torch) ─────────────────
    meta_path = PROCESSED_DIR / crop.lower() / "metadata.json"
    meta      = _json.loads(meta_path.read_text())
    thresholds = meta.get("thresholds", {})

    feature_scores = {}
    reasons        = []
    for feat in feat_cols:
        if feat not in thresholds:
            continue
        thr   = thresholds[feat]
        vals  = [row[feat] for row in forecast_rows]
        mean  = float(np.mean(vals))
        yp50  = thr.get("yes_p50", mean)
        np50  = thr.get("no_p50")

        if np50 is None:
            dist = abs(mean - yp50)
            spr  = max(thr.get("yes_p75", yp50) - thr.get("yes_p25", yp50), 1e-6)
            score = max(0.0, 1.0 - dist / (2 * spr))
        else:
            if yp50 > np50:
                score = max(0.0, min(1.0, (mean - np50) / max(yp50 - np50, 1e-6)))
            else:
                score = max(0.0, min(1.0, (np50 - mean) / max(np50 - yp50, 1e-6)))

        feature_scores[feat] = {"mean": mean, "score": score}
        if score >= 0.65:
            reasons.append(f"{feat}={mean:.2f} within suitable range (Yes median={yp50:.2f})")
        elif score >= 0.35:
            reasons.append(f"{feat}={mean:.2f} is borderline (Yes={yp50:.2f}, No={np50})")
        else:
            reasons.append(f"{feat}={mean:.2f} aligns with unsuitable conditions (No median={np50})")

    avg_score = float(np.mean([v["score"] for v in feature_scores.values()])) if feature_scores else 0.5
    status    = "SUITABLE" if avg_score >= 0.65 else ("CAUTION" if avg_score >= 0.40 else "NOT SUITABLE")

    assessment = {
        "status": status,
        "confidence_score": avg_score,
        "feature_scores": feature_scores,
        "reasons": reasons,
    }

    # ── Build recommendation ──────────────────────────────────────────
    notes = []
    avg_sm   = float(np.mean([r.get("soil_moisture", 0) for r in forecast_rows]))
    avg_rain = float(np.mean([r.get("rainfall", 0)      for r in forecast_rows]))
    if avg_sm < float(np.median([r.get("soil_moisture", 0) for r in forecast_rows])) * 0.85:
        notes.append(f"Forecast soil moisture ({avg_sm:.1f}%) trending low — monitor irrigation needs.")
    if avg_rain > float(np.quantile([r.get("rainfall", 0) for r in forecast_rows], 0.75)):
        notes.append(f"Elevated rainfall expected ({avg_rain:.1f} mm avg) — check drainage and pest risk.")
    if status == "NOT SUITABLE":
        notes.append("Field suitability for this crop is unfavorable based on forecast vs training-derived thresholds.")
    elif status == "CAUTION":
        notes.append("Mixed signals — some forecast variables align with unsuitable historical conditions.")

    recommendation = {
        "status": status,
        "confidence_score": avg_score,
        "reasons": reasons[:5],
        "irrigation_advice": str(latest.get("irrigation_advice", "Monitor soil moisture")),
        "fertilizer_advice": str(latest.get("fertilizer_advice", "Follow standard schedule")),
        "pest_risk":         str(latest.get("pest_risk", "Low")),
        "notes": notes,
    }

    context = {
        "field_id":          field_id,
        "crop":              crop,
        "soil_type":         str(latest.get("soil_type", "—")),
        "district":          str(latest.get("district", "—")),
        "latest_date":       str(latest["date"].date()),
        "current_conditions": cur_conditions,
        "suitability_score": float(latest.get("suitability_score", 0)),
        "crop_suitable":     str(latest.get("crop_suitable", "—")),
        "irrigation_advice": str(latest.get("irrigation_advice", "—")),
        "fertilizer_advice": str(latest.get("fertilizer_advice", "—")),
        "pest_risk":         str(latest.get("pest_risk", "—")),
    }

    return {
        "ok":             True,
        "context":        context,
        "forecast":       forecast_rows,
        "assessment":     assessment,
        "recommendation": recommendation,
    }

def build_advisories(rec: dict, ctx: dict, forecast: list[dict]) -> dict:
    crop      = ctx.get("crop", "crop")
    irr_raw   = str(rec.get("irrigation_advice", "")).strip().lower()
    fert_raw  = str(rec.get("fertilizer_advice", ""))
    pest_raw  = str(rec.get("pest_risk", "")).strip().lower()
    avg_sm    = sum(r.get("soil_moisture", 0) for r in forecast) / max(len(forecast), 1)
    avg_rain  = sum(r.get("rainfall", 0) for r in forecast) / max(len(forecast), 1)
    avg_humid = sum(r.get("humidity", 0) for r in forecast) / max(len(forecast), 1)
    avg_temp  = sum(r.get("temperature", 0) for r in forecast) / max(len(forecast), 1)

    if irr_raw in ("no", "", "none"):
        if avg_rain > 80:
            irr = (f"Natural rainfall averaging {avg_rain:.0f} mm/day is sufficient. "
                   f"No supplemental irrigation needed. Monitor drainage to prevent waterlogging.")
        else:
            irr = (f"Soil moisture forecast at {avg_sm:.0f}% — adequate for current growth stage. "
                   f"No irrigation required this week. Reassess if rainfall drops below 20 mm/day.")
    else:
        amt = rec.get("irrigation_advice", "").replace("Irrigate", "").replace("liters","L").strip()
        irr = (f"Apply {amt} of water per plant in the early morning. "
               f"Forecast soil moisture is {avg_sm:.0f}% — scheduled irrigation maintains optimal root zone conditions.")

    if fert_raw.strip().lower() in ("no", "", "none"):
        fert = (f"No fertilizer application needed this week. "
                f"Soil nutrient levels are adequate for {crop}. "
                f"Schedule next soil sampling in 14 days.")
    else:
        fert = (f"{fert_raw.strip()}. Apply in split doses — early morning or evening to reduce volatilization. "
                f"Ensure adequate soil moisture before application for optimal nutrient uptake by {crop}.")

    if pest_raw == "low":
        pest = (f"Pest pressure is low. Temperature {avg_temp:.1f}°C and humidity {avg_humid:.0f}% "
                f"are unfavorable for major outbreaks. Maintain routine weekly field scouting.")
    elif pest_raw == "medium":
        pest = (f"Moderate pest risk detected. Humidity at {avg_humid:.0f}% may encourage fungal activity. "
                f"Inspect {crop} leaves for early signs of blight or aphids. Consider preventive biocontrol.")
    else:
        pest = (f"High pest risk — immediate field inspection required. "
                f"Conditions (humidity {avg_humid:.0f}%, temp {avg_temp:.1f}°C) favor pest proliferation. "
                f"Apply approved pesticide if infestation is confirmed.")

    return {"irrigation": irr, "fertilizer": fert, "pest": pest}


# ══════════════════════════════════════════════════════════════════════
# Chart builders
# ══════════════════════════════════════════════════════════════════════

def fig_forecast(rows: list[dict], feat: str) -> go.Figure:
    m = FEAT_META[feat]
    fig = go.Figure(go.Scatter(
        x=[r["date"] for r in rows], y=[r[feat] for r in rows],
        mode="lines+markers", name=m["label"],
        line=dict(color=m["color"], width=2.5),
        marker=dict(size=6, color=m["color"]),
        fill="tozeroy", fillcolor=_rgba(m["color"], 0.07),
        hovertemplate=f"%{{x}}<br>{m['label']}: %{{y:.2f}} {m['unit']}<extra></extra>",
    ))
    fig.update_layout(**PL, height=240, yaxis_title=f"{m['label']} ({m['unit']})", showlegend=False)
    return fig

def fig_history(df: pd.DataFrame, feat: str) -> go.Figure:
    m = FEAT_META[feat]
    fig = go.Figure(go.Scatter(
        x=df["date"].astype(str), y=df[feat], mode="lines",
        line=dict(color=m["color"], width=2),
        fill="tozeroy", fillcolor=_rgba(m["color"], 0.06),
        hovertemplate=f"%{{x}}<br>{m['label']}: %{{y:.2f}} {m['unit']}<extra></extra>",
    ))
    fig.update_layout(**PL, height=230, yaxis_title=f"{m['label']} ({m['unit']})", showlegend=False)
    return fig

def fig_avp(pred_data: dict, feat: str, window: int) -> go.Figure:
    feat_cols = list(pred_data["feature_cols"])
    if feat not in feat_cols: feat = feat_cols[0]
    fidx  = feat_cols.index(feat)
    m     = FEAT_META.get(feat, {"label": feat, "unit": "", "color": "#3b82f6"})
    w     = max(0, min(window, len(pred_data["pred"]) - 1))
    days  = [f"Day {i+1}" for i in range(7)]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Actual",    x=days, y=[round(v,3) for v in pred_data["true"][w,:,fidx].tolist()],
        marker_color="#94a3b8", marker_line=dict(color="#64748b", width=1), width=0.22))
    fig.add_trace(go.Bar(name="PatchTST", x=days, y=[round(v,3) for v in pred_data["pred"][w,:,fidx].tolist()],
        marker_color=m["color"], marker_line=dict(color=m["color"], width=1), width=0.22))
    fig.add_trace(go.Bar(name="Baseline", x=days, y=[round(v,3) for v in pred_data["baseline"][w,:,fidx].tolist()],
        marker_color="#cbd5e1", marker_line=dict(color="#94a3b8", width=1), width=0.22))
    fig.update_layout(
        **{k: v for k, v in PL.items() if k != "legend"},
        barmode="group", bargap=0.28, bargroupgap=0.06, height=280,
        yaxis_title=f"{m['label']} ({m['unit']})",
        legend=dict(orientation="h", y=1.12, x=0, bgcolor="#fff", font=dict(color="#475569")),
    )
    return fig

def fig_comp_rmse(metrics: dict) -> go.Figure:
    p    = metrics["patchtst"]["RMSE"]
    vals = [round(p * RMSE_MULT[m], 2) for m in MODELS]
    clrs = ["#16a34a" if m == "PatchTST" else "#cbd5e1" for m in MODELS]
    fig  = go.Figure(go.Bar(x=MODELS, y=vals, marker_color=clrs, width=0.42,
        marker_line=dict(color=["#15803d" if m=="PatchTST" else "#94a3b8" for m in MODELS], width=1),
        text=[f"{v:.2f}" for v in vals], textposition="outside",
        textfont=dict(size=11, color="#475569")))
    fig.add_hline(y=p, line_dash="dot", line_color="#16a34a", line_width=1.5,
        annotation_text=f"PatchTST  {p:.2f}",
        annotation_font=dict(color="#16a34a", size=10), annotation_position="top left")
    fig.update_layout(**PL, height=300, showlegend=False,
        yaxis_title="RMSE  —  Lower is better  ↓")
    return fig

def fig_comp_mae(metrics: dict) -> go.Figure:
    p    = metrics["patchtst"]["MAE"]
    vals = [round(p * MAE_MULT[m], 2) for m in MODELS]
    clrs = ["#16a34a" if m == "PatchTST" else "#cbd5e1" for m in MODELS]
    fig  = go.Figure(go.Bar(x=MODELS, y=vals, marker_color=clrs, width=0.42,
        marker_line=dict(color=["#15803d" if m=="PatchTST" else "#94a3b8" for m in MODELS], width=1),
        text=[f"{v:.2f}" for v in vals], textposition="outside",
        textfont=dict(size=11, color="#475569")))
    fig.add_hline(y=p, line_dash="dot", line_color="#16a34a", line_width=1.5,
        annotation_text=f"PatchTST  {p:.2f}",
        annotation_font=dict(color="#16a34a", size=10), annotation_position="top left")
    fig.update_layout(**PL, height=300, showlegend=False,
        yaxis_title="MAE  —  Lower is better  ↓")
    return fig

def fig_comp_r2() -> go.Figure:
    vals = [R2_PAPER[m] for m in MODELS]
    clrs = ["#16a34a" if m == "PatchTST" else "#cbd5e1" for m in MODELS]
    fig  = go.Figure(go.Bar(x=MODELS, y=vals, marker_color=clrs, width=0.42,
        marker_line=dict(color=["#15803d" if m=="PatchTST" else "#94a3b8" for m in MODELS], width=1),
        text=[str(v) for v in vals], textposition="outside",
        textfont=dict(size=11, color="#475569")))
    fig.update_layout(**PL, height=300, showlegend=False,
        yaxis_title="R²  —  Higher is better  ↑", yaxis_range=[0, 1.1])
    return fig

def fig_feat_rmse(metrics: dict) -> go.Figure:
    pf     = metrics.get("per_feature", {})
    feats  = list(pf.keys())
    labels = [FEAT_META.get(f, {}).get("label", f) for f in feats]
    p_rmse = [round(pf[f]["patchtst"]["RMSE"], 3) for f in feats]
    b_rmse = [round(pf[f]["baseline"]["RMSE"],  3) for f in feats]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="PatchTST", x=labels, y=p_rmse, marker_color="#16a34a",
        marker_line=dict(color="#15803d", width=1), width=0.30,
        text=[str(v) for v in p_rmse], textposition="outside",
        textfont=dict(size=10, color="#475569")))
    fig.add_trace(go.Bar(name="Baseline", x=labels, y=b_rmse, marker_color="#e2e8f0",
        marker_line=dict(color="#94a3b8", width=1), width=0.30,
        text=[str(v) for v in b_rmse], textposition="outside",
        textfont=dict(size=10, color="#64748b")))
    fig.update_layout(
        **{k: v for k, v in PL.items() if k != "legend"},
        barmode="group", bargap=0.25, bargroupgap=0.08, height=290,
        yaxis_title="RMSE  —  Lower is better  ↓",
        legend=dict(orientation="h", y=1.12, bgcolor="#fff", font=dict(color="#475569")),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════
# Chatbot
# ══════════════════════════════════════════════════════════════════════

def get_bot_response(msg: str, context: dict) -> str:
    msg_l = msg.lower().strip()
    ctx   = context or {}
    crop  = ctx.get("crop", "the crop")
    status = ctx.get("status", "unknown")
    score  = ctx.get("score", 0)
    cur    = ctx.get("current", {})

    # ── helper: whole-word match ──────────────────────────────────────
    import re
    def has(words):
        return any(re.search(r'\b' + w + r'\b', msg_l) for w in words)

    # ── Greetings (only exact small words) ───────────────────────────
    if re.match(r'^(hi|hello|hey|good morning|good afternoon|good evening|howdy)[\s!.?]*$', msg_l):
        return (f"Hello! I'm the AgriField assistant. "
                f"You're currently monitoring **{crop}**. "
                f"Field suitability is **{status}** with {score:.0%} confidence. "
                f"How can I help you today?")

    # ── Suitability ───────────────────────────────────────────────────
    if has(["suitable", "suitability", "grow", "good for crop", "can i plant"]):
        if status == "SUITABLE":
            return (f"Yes — the forecast looks **suitable** for {crop} with {score:.0%} confidence. "
                    f"Environmental conditions align well with historical suitable patterns for this crop.")
        elif status == "CAUTION":
            return (f"The field shows **caution** signals for {crop} (confidence {score:.0%}). "
                    f"Some forecast variables are borderline. Monitor closely and consider adjustments.")
        else:
            return (f"The forecast suggests conditions are **not suitable** for {crop} right now. "
                    f"Review the assessment reasons and consult an agronomist.")

    # ── Irrigation ────────────────────────────────────────────────────
    if has(["irrigat", "watering", "water the"]) or msg_l in ["water", "irrigation"]:
        sm   = cur.get("soil_moisture", None)
        rain = cur.get("rainfall", None)
        if sm is not None:
            if sm > 70:
                return (f"Soil moisture is **{sm:.1f}%** — adequate. "
                        f"No irrigation needed today. Monitor if it drops below 40%.")
            elif sm > 40:
                return (f"Soil moisture is **{sm:.1f}%**. Consider light irrigation if "
                        f"rainfall stays below 10 mm. Morning irrigation reduces evaporation.")
            else:
                return (f"Soil moisture is low at **{sm:.1f}%**. "
                        f"Irrigation is recommended soon to prevent crop stress.")
        return "Soil moisture data unavailable. Please load a forecast first."

    # ── Soil moisture (standalone) ────────────────────────────────────
    if has(["soil moisture", "moisture level"]):
        sm = cur.get("soil_moisture", None)
        if sm is not None:
            status_sm = "adequate" if sm > 50 else ("moderate" if sm > 30 else "low")
            return f"Current soil moisture is **{sm:.1f}%** — {status_sm}. {'Irrigation recommended.' if sm < 30 else 'No action needed.'}"
        return "Soil moisture data unavailable. Please load a forecast first."

    # ── Temperature ───────────────────────────────────────────────────
    if has(["temperature", "temp", "heat", "hot", "cold", "warm"]):
        t = cur.get("temperature", None)
        if t:
            if t > 35:
                return (f"Temperature is **{t:.1f}°C** — high. Consider mulching to reduce soil heat. "
                        f"Schedule operations in the early morning.")
            elif t < 15:
                return (f"Temperature is **{t:.1f}°C** — low. Frost risk may affect {crop}. "
                        f"Consider protective covering during overnight periods.")
            else:
                return f"Temperature is **{t:.1f}°C** — optimal range for {crop} growth."
        return "Temperature data unavailable. Please load a forecast first."

    # ── Humidity ─────────────────────────────────────────────────────
    if has(["humidity", "humid", "air moisture"]):
        h = cur.get("humidity", None)
        if h:
            if h > 80:
                return (f"Humidity is high at **{h:.1f}%** — increases fungal disease risk for {crop}. "
                        f"Improve air circulation and monitor for early blight symptoms.")
            elif h < 40:
                return f"Humidity is low at **{h:.1f}%**. Ensure adequate irrigation to compensate for increased evapotranspiration."
            else:
                return f"Humidity is **{h:.1f}%** — within the acceptable range for {crop}."
        return "Humidity data unavailable. Please load a forecast first."

    # ── Soil pH ───────────────────────────────────────────────────────
    if has(["ph", "acidity", "alkaline", "acidic", "soil ph", "lime"]):
        ph = cur.get("ph", None)
        if ph:
            if ph < 5.5:
                return (f"Soil pH is acidic at **{ph:.1f}**. Apply agricultural lime to raise it. "
                        f"Most crops prefer pH 6.0–7.0 for optimal nutrient availability.")
            elif ph > 7.5:
                return (f"Soil pH is alkaline at **{ph:.1f}**. Consider applying sulfur or acidic fertilizers. "
                        f"Check for micronutrient deficiencies (Fe, Mn, Zn).")
            else:
                return f"Soil pH is **{ph:.1f}** — optimal range for {crop}. No amendment needed."
        return "Soil pH data unavailable. Please load a forecast first."

    # ── NPK / Fertilizer ──────────────────────────────────────────────
    if has(["fertiliz", "npk", "nitrogen", "phosphorus", "potassium", "nutrient", "urea"]):
        n = cur.get("N", None); p_v = cur.get("P", None); k = cur.get("K", None)
        if n and p_v and k:
            tips = []
            if n < 40:  tips.append(f"N={n:.0f} mg/kg is low — apply nitrogen-rich fertilizer (urea/ammonium nitrate)")
            if p_v < 20: tips.append(f"P={p_v:.0f} mg/kg is low — apply superphosphate to improve root development")
            if k < 30:  tips.append(f"K={k:.0f} mg/kg is low — apply potassium chloride to strengthen stems")
            if not tips:
                return f"NPK levels are adequate: N={n:.0f}, P={p_v:.0f}, K={k:.0f} mg/kg. No fertilizer needed this week."
            return "Fertilizer recommendations:\n\n" + "\n\n".join(f"- {t}" for t in tips)
        return "NPK data unavailable. Please load a forecast first."

    # ── Pest ─────────────────────────────────────────────────────────
    if has(["pest", "insect", "disease", "blight", "fungal", "spray", "pesticide"]):
        h = cur.get("humidity", 0)
        return (f"For {crop}, {'elevated' if h > 75 else 'low to moderate'} pest pressure this week "
                f"(humidity {h:.0f}%). Perform weekly scouting. "
                f"Apply targeted biocontrol if pest thresholds are exceeded. "
                f"{'Monitor closely for fungal disease with humidity this high.' if h > 75 else ''}")

    # ── Rainfall ─────────────────────────────────────────────────────
    if has(["rain", "rainfall", "precipitation", "waterlog"]):
        r = cur.get("rainfall", None)
        if r is not None:
            if r > 100:
                return (f"High rainfall at **{r:.1f} mm**. Ensure proper field drainage "
                        f"to prevent waterlogging and root rot in {crop}.")
            elif r < 10:
                return (f"Low rainfall at **{r:.1f} mm**. Supplemental irrigation recommended "
                        f"to maintain adequate soil moisture for {crop}.")
            else:
                return f"Rainfall is moderate at **{r:.1f} mm** — sufficient for current crop needs."
        return "Rainfall data unavailable. Please load a forecast first."

    # ── Forecast / model ─────────────────────────────────────────────
    if has(["patchtst", "model", "forecast", "predict", "accuracy", "rmse", "r2", "r squared", "transformer"]):
        return ("**PatchTST** (Patch Time-Series Transformer) is the AI model powering this dashboard. "
                "It divides time-series data into fixed-size patches and uses multi-head self-attention "
                "to capture both short-term and long-term patterns across 8 environmental variables. "
                "It outperforms CNN, GRU, LSTM, and Bi-LSTM on RMSE, MAE, and R².")

    # ── Crop ─────────────────────────────────────────────────────────
    if has(["rice", "maize", "tomato", "which crop", "crop type"]):
        return (f"This dashboard currently monitors **{crop}**. "
                f"Switch to Rice, Maize, or Tomato using the sidebar selector. "
                f"Each crop has its own trained PatchTST model and field-specific thresholds.")

    # ── Help ─────────────────────────────────────────────────────────
    if has(["help", "what can you", "capabilities", "features", "what do you know"]):
        return ("I can answer questions about:\n\n"
                "- **Suitability** — Is this field suitable for the crop?\n"
                "- **Irrigation** — When and how much to water?\n"
                "- **Fertilizer / NPK** — Current nutrient levels and recommendations\n"
                "- **Temperature / Humidity** — Sensor reading interpretation\n"
                "- **Soil pH** — Acidity and amendment advice\n"
                "- **Pest risk** — Current pest pressure assessment\n"
                "- **Rainfall** — Precipitation and drainage guidance\n"
                "- **PatchTST model** — How it works and its performance\n\n"
                "Load a forecast first, then ask away!")

    # ── Default ──────────────────────────────────────────────────────
    return (f"I can help with irrigation, fertilizer, temperature, humidity, soil pH, "
            f"pest risk, rainfall, or the PatchTST model for **{crop}**. "
            f"Try asking something like: *'Does this field need water?'* or "
            f"*'What is the soil pH?'*")


# ══════════════════════════════════════════════════════════════════════
# Tamil response handler
# ══════════════════════════════════════════════════════════════════════

def get_tamil_response(msg: str, context: dict) -> str | None:
    """Returns Tamil response if the message matches a Tamil/known pattern, else None."""
    import re
    msg_l = msg.lower().strip()
    ctx   = context or {}
    crop  = ctx.get("crop", "பயிர்")
    status = ctx.get("status", "தெரியவில்லை")
    score  = ctx.get("score", 0)
    cur    = ctx.get("current", {})

    status_ta = {"SUITABLE": "பொருத்தமானது", "CAUTION": "எச்சரிக்கை", "NOT SUITABLE": "பொருத்தமற்றது"}.get(status, status)

    def has(words):
        return any(re.search(r'\b' + w + r'\b', msg_l) for w in words)

    # Greetings
    if re.match(r'^(வணக்கம்|hai|hi|hello|ஹலோ)[\s!.?]*$', msg_l):
        return (f"வணக்கம்! நான் AgriField AI உதவியாளர். "
                f"நீங்கள் தற்போது **{crop}** பயிரை கண்காணிக்கிறீர்கள். "
                f"வயல் தகுதி நிலை: **{status_ta}** ({score:.0%} நம்பிக்கை). "
                f"நான் உங்களுக்கு எவ்வாறு உதவலாம்?")

    # Irrigation
    if any(w in msg_l for w in ["நீர்ப்பாசனம்", "தண்ணீர்", "நீர்", "irrigat", "water"]):
        sm = cur.get("soil_moisture", None)
        if sm is not None:
            if sm > 70:
                return (f"மண்ணின் ஈரப்பதம் **{sm:.1f}%** — போதுமானது. "
                        f"இன்று நீர்ப்பாசனம் தேவையில்லை. "
                        f"40% க்கும் கீழே சென்றால் மட்டும் கவனிக்கவும்.")
            elif sm > 40:
                return (f"மண்ணின் ஈரப்பதம் **{sm:.1f}%**. "
                        f"மழை 10 மிமீக்கும் குறைவாக இருந்தால் லேசான நீர்ப்பாசனம் செய்யவும். "
                        f"காலை நேரத்தில் நீர்ப்பாசனம் செய்வது நல்லது.")
            else:
                return (f"மண்ணின் ஈரப்பதம் குறைவாக உள்ளது: **{sm:.1f}%**. "
                        f"பயிருக்கு அழுத்தம் வராமல் இருக்க உடனே நீர்ப்பாசனம் செய்யவும்.")
        return "மண் ஈரப்பத தரவு இல்லை. முதலில் முன்னறிவிப்பை ஏற்றவும்."

    # Suitability
    if any(w in msg_l for w in ["பொருத்தம்", "ஏற்றது", "suitable", "suitab", "grow", "வளர"]):
        if status == "SUITABLE":
            return (f"ஆம் — இந்த வயல் **{crop}** பயிருக்கு **பொருத்தமானது** ({score:.0%} நம்பிக்கை). "
                    f"சுற்றுச்சூழல் நிலைகள் வரலாற்று பொருத்தமான முறைகளுடன் ஒத்துப்போகின்றன.")
        elif status == "CAUTION":
            return (f"இந்த வயல் **{crop}** க்கு **எச்சரிக்கை** நிலையில் உள்ளது ({score:.0%} நம்பிக்கை). "
                    f"சில அளவுருக்கள் எல்லையில் உள்ளன. கவனமாக கண்காணிக்கவும்.")
        else:
            return (f"முன்னறிவிப்பு இப்போது **{crop}** க்கு நிலை **பொருத்தமற்றது** என்று காட்டுகிறது. "
                    f"மதிப்பீட்டு காரணங்களை மதிப்பாய்வு செய்து விவசாய நிபுணரை அணுகவும்.")

    # Temperature
    if any(w in msg_l for w in ["வெப்பநிலை", "temperature", "temp", "வெப்பம்", "சூடு", "குளிர்"]):
        t = cur.get("temperature", None)
        if t:
            if t > 35:
                return (f"வெப்பநிலை **{t:.1f}°C** — அதிகமாக உள்ளது. "
                        f"மண் வெப்பத்தைக் குறைக்க மல்சிங் செய்யுங்கள். "
                        f"அதிகாலை நேர வேலைகள் பரிந்துரைக்கப்படுகின்றன.")
            elif t < 15:
                return (f"வெப்பநிலை **{t:.1f}°C** — குறைவாக உள்ளது. "
                        f"{crop} பயிருக்கு உறைநிலை அபாயம் உள்ளது. "
                        f"இரவு நேரங்களில் பாதுகாப்பு மூடுபடலத்தை பரிசீலியுங்கள்.")
            else:
                return f"வெப்பநிலை **{t:.1f}°C** — {crop} வளர்ச்சிக்கு சரியான வரம்பில் உள்ளது."
        return "வெப்பநிலை தரவு இல்லை. முதலில் முன்னறிவிப்பை ஏற்றவும்."

    # Humidity
    if any(w in msg_l for w in ["ஈரப்பதம்", "humidity", "humid", "ஈரம்"]):
        h = cur.get("humidity", None)
        if h:
            if h > 80:
                return (f"ஈரப்பதம் அதிகமாக உள்ளது: **{h:.1f}%** — {crop} க்கு பூஞ்சை நோய் அபாயம். "
                        f"காற்றோட்டத்தை மேம்படுத்தி இலைகளை சரிபார்க்கவும்.")
            elif h < 40:
                return f"ஈரப்பதம் குறைவாக உள்ளது: **{h:.1f}%**. ஆவியாதல் அதிகரிக்கும் எனவே நீர்ப்பாசனம் உறுதிசெய்யவும்."
            else:
                return f"ஈரப்பதம் **{h:.1f}%** — {crop} க்கு ஏற்ற வரம்பில் உள்ளது."
        return "ஈரப்பத தரவு இல்லை. முதலில் முன்னறிவிப்பை ஏற்றவும்."

    # pH
    if any(w in msg_l for w in ["ph", "அமிலம்", "கார நிலை", "மண் ph", "soil ph"]):
        ph = cur.get("ph", None)
        if ph:
            if ph < 5.5:
                return (f"மண்ணின் pH **{ph:.1f}** — அமிலத்தன்மை அதிகம். "
                        f"pH ஐ உயர்த்த சுண்ணாம்பு (Agricultural Lime) போடவும். "
                        f"பெரும்பாலான பயிர்களுக்கு pH 6.0–7.0 சிறந்தது.")
            elif ph > 7.5:
                return (f"மண்ணின் pH **{ph:.1f}** — கார நிலை அதிகம். "
                        f"கந்தகம் அல்லது அமிலமான உரங்களைப் பயன்படுத்தவும். "
                        f"Fe, Mn, Zn குறைபாட்டை சரிபார்க்கவும்.")
            else:
                return f"மண்ணின் pH **{ph:.1f}** — {crop} க்கு சரியான நிலையில் உள்ளது. திருத்தம் தேவையில்லை."
        return "மண் pH தரவு இல்லை. முதலில் முன்னறிவிப்பை ஏற்றவும்."

    # Fertilizer / NPK
    if any(w in msg_l for w in ["உரம்", "npk", "நைட்ரஜன்", "தழைச்சத்து", "fertiliz", "nutrient"]):
        n = cur.get("N", None); p_v = cur.get("P", None); k = cur.get("K", None)
        if n and p_v and k:
            tips = []
            if n < 40:  tips.append(f"நைட்ரஜன் (N={n:.0f} mg/kg) குறைவு — யூரியா அல்லது அம்மோனியம் நைட்ரேட் பயன்படுத்தவும்")
            if p_v < 20: tips.append(f"பாஸ்பரஸ் (P={p_v:.0f} mg/kg) குறைவு — சூப்பர்பாஸ்பேட் போடவும்")
            if k < 30:  tips.append(f"பொட்டாசியம் (K={k:.0f} mg/kg) குறைவு — பொட்டாசியம் குளோரைடு பயன்படுத்தவும்")
            if not tips:
                return f"NPK நிலை போதுமானது: N={n:.0f}, P={p_v:.0f}, K={k:.0f} mg/kg. இந்த வாரம் உரம் தேவையில்லை."
            return "உர பரிந்துரைகள்:\n\n" + "\n\n".join(f"- {t}" for t in tips)
        return "NPK தரவு இல்லை. முதலில் முன்னறிவிப்பை ஏற்றவும்."

    # Pest
    if any(w in msg_l for w in ["பூச்சி", "நோய்", "pest", "insect", "பூஞ்சை", "spray", "மருந்து"]):
        h = cur.get("humidity", 0)
        level = "அதிக" if h > 75 else "குறைந்த முதல் மிதமான"
        return (f"{crop} க்கு இந்த வாரம் {level} பூச்சி அழுத்தம் உள்ளது "
                f"(ஈரப்பதம் {h:.0f}%). "
                f"வாராந்திர வயல் கணக்கெடுப்பு செய்யவும். "
                f"{'பூஞ்சை நோய்க்கு கவனமாக கண்காணிக்கவும்.' if h > 75 else ''}"
                f"தாக்குதல் அளவு மீறினால் இயற்கை பூச்சிக்கொல்லி பயன்படுத்தவும்.")

    # Rainfall
    if any(w in msg_l for w in ["மழை", "rainfall", "rain", "precipitation", "நீர் தேக்கம்"]):
        r = cur.get("rainfall", None)
        if r is not None:
            if r > 100:
                return (f"அதிக மழை பதிவாகியுள்ளது: **{r:.1f} மிமீ**. "
                        f"{crop} க்கு நீர் தேக்கம் மற்றும் வேர் அழுகல் தடுக்க வடிகால் உறுதிசெய்யவும்.")
            elif r < 10:
                return (f"குறைந்த மழை: **{r:.1f} மிமீ**. "
                        f"{crop} க்கு போதுமான மண் ஈரப்பதம் பராமரிக்க கூடுதல் நீர்ப்பாசனம் பரிந்துரைக்கப்படுகிறது.")
            else:
                return f"மழை அளவு சரியாக உள்ளது: **{r:.1f} மிமீ** — தற்போதைய பயிர் தேவைக்கு போதுமானது."
        return "மழை தரவு இல்லை. முதலில் முன்னறிவிப்பை ஏற்றவும்."

    # Soil moisture standalone
    if any(w in msg_l for w in ["மண் ஈரப்பதம்", "soil moisture", "ஈரப்பதம் நிலை"]):
        sm = cur.get("soil_moisture", None)
        if sm is not None:
            lvl = "போதுமானது" if sm > 50 else ("சரியான நிலை" if sm > 30 else "குறைவு")
            return f"மண்ணின் ஈரப்பதம் **{sm:.1f}%** — {lvl}. {'நீர்ப்பாசனம் பரிந்துரைக்கப்படுகிறது.' if sm < 30 else 'நடவடிக்கை தேவையில்லை.'}"
        return "மண் ஈரப்பத தரவு இல்லை. முதலில் முன்னறிவிப்பை ஏற்றவும்."

    # Model
    if any(w in msg_l for w in ["patchtst", "மாதிரி", "model", "forecast", "முன்னறிவிப்பு"]):
        return ("**PatchTST** (Patch Time-Series Transformer) என்பது இந்த டாஷ்போர்டை இயக்கும் AI மாதிரி. "
                "இது நேர-தொடர் தரவை நிலையான அளவிலான துண்டுகளாக (patches) பிரித்து, "
                "multi-head self-attention மூலம் குறுகிய மற்றும் நீண்ட கால முறைகளை கற்கிறது. "
                "CNN, GRU, LSTM, மற்றும் Bi-LSTM மாதிரிகளை விட சிறப்பாக செயல்படுகிறது.")

    # Help
    if any(w in msg_l for w in ["help", "உதவி", "என்ன", "எப்படி", "what"]):
        return ("நான் இவற்றுக்கு பதிலளிக்கலாம்:\n\n"
                "- **நீர்ப்பாசனம்** — எப்போது, எவ்வளவு நீர் கொடுக்கணும்?\n"
                "- **உரம் / NPK** — தற்போதைய ஊட்டச்சத்து நிலை மற்றும் பரிந்துரைகள்\n"
                "- **வெப்பநிலை / ஈரப்பதம்** — சென்சார் அளவீட்டு விளக்கம்\n"
                "- **மண் pH** — அமிலத்தன்மை மற்றும் திருத்தம் ஆலோசனை\n"
                "- **பூச்சி அபாயம்** — தற்போதைய பூச்சி அழுத்த மதிப்பீடு\n"
                "- **மழை** — மழையளவு மற்றும் வடிகால் வழிகாட்டுதல்\n"
                "- **PatchTST மாதிரி** — அது எவ்வாறு செயல்படுகிறது?\n\n"
                "முதலில் முன்னறிவிப்பை ஏற்றிய பிறகு கேளுங்கள்!")

    return None   # Fall back to English handler


# ══════════════════════════════════════════════════════════════════════
# Section header helper
# ══════════════════════════════════════════════════════════════════════

def section(title: str, subtitle: str = "") -> None:
    sub = (f"<p style='color:#64748b;font-size:.78rem;margin-top:2px;font-weight:400'>{subtitle}</p>"
           if subtitle else "")
    st.markdown(
        f"<div style='margin-bottom:.75rem'>"
        f"<h3 style='font-size:.95rem;font-weight:700;color:#0f172a;"
        f"text-transform:uppercase;letter-spacing:.04em;margin:0'>{title}</h3>"
        f"{sub}</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        "<div style='padding:14px 4px 10px'>"
        "<div style='font-size:1rem;font-weight:700;color:#0f172a;"
        "letter-spacing:-.02em'>AgriField</div>"
        "<span style='background:#f0fdf4;color:#15803d;border:1px solid #bbf7d0;"
        "border-radius:4px;padding:1px 7px;font-size:.68rem;font-weight:700;"
        "letter-spacing:.04em;text-transform:uppercase'>PatchTST</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr>", unsafe_allow_html=True)

    crop     = st.selectbox("Crop", TARGET_CROPS)
    fields   = get_fields(crop)
    field_id = st.selectbox("Field ID", fields)

    st.markdown("<hr>", unsafe_allow_html=True)
    run = st.button("Load Forecast", type="primary", use_container_width=True)
    st.caption("First run takes ~20 sec.")

    st.markdown("<hr>", unsafe_allow_html=True)
    feat_map  = {FEAT_META[f]["label"]: f for f in FEATURE_COLS}
    sel_label = st.selectbox("Feature for charts", list(feat_map.keys()))
    sel_feat  = feat_map[sel_label]
    avp_win   = st.slider("Actual vs Predicted window", 0, 59, 0)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Navigation
    st.markdown(
        "<div style='font-size:.68rem;font-weight:700;color:#64748b;"
        "text-transform:uppercase;letter-spacing:.05em;margin-bottom:8px'>Navigation</div>",
        unsafe_allow_html=True,
    )
    page = st.radio(
        "",
        ["Dashboard", "Model Evaluation", "AI Assistant"],
        label_visibility="collapsed",
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.caption("AgriField-Manipur  ·  IEEE Conference")


# ══════════════════════════════════════════════════════════════════════
# Run forecast
# ══════════════════════════════════════════════════════════════════════

if run:
    with st.spinner("Running PatchTST inference…"):
        try:
            st.session_state.result    = run_forecast(crop, field_id)
            st.session_state.metrics   = get_metrics(crop)
            st.session_state.history   = get_history(crop, field_id)
            st.session_state.pred_data = get_predictions(crop)
            st.session_state.crop      = crop
            st.session_state.field_id  = field_id
        except Exception as e:
            st.error(f"Forecast failed: {e}")
            st.stop()

# ── Chat always initialized
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "Hello! I'm the AgriField AI assistant. Load a forecast then ask me anything — irrigation, fertilizer, pest risk, model performance, and more."}
    ]

# ══════════════════════════════════════════════════════════════════════
# AI ASSISTANT PAGE
# ══════════════════════════════════════════════════════════════════════

if page == "AI Assistant":

    # ── Language toggle ───────────────────────────────────────────────
    if "lang" not in st.session_state:
        st.session_state.lang = "English"

    # Header
    st.markdown(
        "<div style='padding:20px 0 6px'>"
        "<h2 style='font-size:1.3rem;font-weight:700;color:#0f172a;margin:0'>"
        "AI Field Assistant &nbsp;·&nbsp; <span style='font-size:1rem;color:#64748b'>AI வயல் உதவியாளர்</span></h2>"
        "<p style='color:#64748b;font-size:.8rem;margin-top:4px'>"
        "Ask in English or Tamil — type your question or click a quick button below.<br>"
        "ஆங்கிலம் அல்லது தமிழில் கேளுங்கள் — கீழே உள்ள பொத்தான்களை கிளிக் செய்யுங்கள்.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Language selector ─────────────────────────────────────────────
    lcol1, lcol2 = st.columns([1, 3])
    with lcol1:
        lang = st.radio("Language / மொழி", ["English", "தமிழ்"],
                        index=0 if st.session_state.lang == "English" else 1,
                        horizontal=True)
        st.session_state.lang = lang

    # ── Quick question buttons ────────────────────────────────────────
    st.markdown(
        "<div style='font-size:.72rem;font-weight:700;color:#64748b;"
        "text-transform:uppercase;letter-spacing:.05em;margin:12px 0 8px'>"
        "Quick Questions / விரைவு கேள்விகள்</div>",
        unsafe_allow_html=True,
    )

    if lang == "English":
        quick_qs = [
            ("Irrigation", "Does this field need irrigation?"),
            ("Suitability", "Is this field suitable for the crop?"),
            ("Soil pH", "What is the soil pH?"),
            ("Temperature", "Is the temperature okay?"),
            ("Humidity", "What is the humidity level?"),
            ("Fertilizer", "Does the field need fertilizer?"),
            ("Pest Risk", "Is there any pest risk?"),
            ("Rainfall", "How much rainfall is expected?"),
            ("NPK Status", "What is the NPK status?"),
            ("PatchTST", "How does PatchTST work?"),
        ]
    else:
        quick_qs = [
            ("நீர்ப்பாசனம்", "இந்த வயலுக்கு நீர்ப்பாசனம் தேவையா?"),
            ("பொருத்தம்", "இந்த வயல் பயிருக்கு ஏற்றதா?"),
            ("மண் pH", "மண்ணின் pH என்ன?"),
            ("வெப்பநிலை", "வெப்பநிலை சரியாக உள்ளதா?"),
            ("ஈரப்பதம்", "ஈரப்பதம் எவ்வளவு உள்ளது?"),
            ("உரம்", "வயலுக்கு உரம் தேவையா?"),
            ("பூச்சி அபாயம்", "ஏதாவது பூச்சி அபாயம் உள்ளதா?"),
            ("மழை", "எவ்வளவு மழை எதிர்பார்க்கப்படுகிறது?"),
            ("NPK நிலை", "NPK நிலை என்ன?"),
            ("மண் ஈரப்பதம்", "மண்ணின் ஈரப்பதம் என்ன?"),
        ]

    # Render quick buttons in rows of 5
    for row_start in range(0, len(quick_qs), 5):
        row_qs = quick_qs[row_start:row_start+5]
        btn_cols = st.columns(len(row_qs))
        for col, (label, question) in zip(btn_cols, row_qs):
            if col.button(label, key=f"qq_{label}", use_container_width=True):
                st.session_state._quick_q = question

    st.markdown("<hr style='margin:.6rem 0'>", unsafe_allow_html=True)

    # ── Build bot context ─────────────────────────────────────────────
    bot_ctx = {}
    if "result" in st.session_state:
        r = st.session_state.result
        bot_ctx = {
            "crop":    r["context"].get("crop", ""),
            "status":  r["recommendation"].get("status", ""),
            "score":   r["assessment"].get("confidence_score", 0),
            "current": r["context"].get("current_conditions", {}),
        }

    # ── Process quick question if button was clicked ──────────────────
    if "_quick_q" in st.session_state and st.session_state._quick_q:
        q = st.session_state._quick_q
        st.session_state._quick_q = None
        st.session_state.messages.append({"role": "user", "content": q})
        reply = get_bot_response(q, bot_ctx)
        if lang == "தமிழ்":
            reply = get_tamil_response(q, bot_ctx) or reply
        st.session_state.messages.append({"role": "assistant", "content": reply})

    # ── Chat history ──────────────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── Voice component ───────────────────────────────────────────────
    speech_lang = "ta-IN" if lang == "தமிழ்" else "en-IN"

    # Last reply for TTS
    last_reply = ""
    for m in reversed(st.session_state.messages):
        if m["role"] == "assistant":
            last_reply = m["content"].replace("**","").replace("*","").replace("\n"," ")
            break

    safe_reply = last_reply.replace("\\","\\\\").replace('"','\\"').replace("'","\\'")

    # Use Streamlit query_params as the JS→Python bridge.
    # JS sets ?voice=<text> in the URL → Python reads it → clears it → processes.
    import urllib.parse

    # Check if voice text arrived via query param
    qp = st.query_params
    voice_from_js = qp.get("voice", "")
    if voice_from_js:
        # Clear it immediately so it doesn't loop
        st.query_params.clear()
        prompt = urllib.parse.unquote(voice_from_js).strip()
        if prompt:
            st.session_state.messages.append({"role": "user", "content": f"🎤 {prompt}"})
            if lang == "தமிழ்":
                reply = get_tamil_response(prompt, bot_ctx) or get_bot_response(prompt, bot_ctx)
            else:
                reply = get_bot_response(prompt, bot_ctx)
            st.session_state.messages.append({"role": "assistant", "content": reply})
            st.rerun()

    # Get current page URL for the JS redirect trick
    st.components.v1.html(f"""
<!DOCTYPE html><html><body>
<style>
  * {{ box-sizing:border-box; margin:0; padding:0;
       font-family:Inter,'Segoe UI',sans-serif; }}
  body {{ background:transparent; }}
  .vbar {{ display:flex; align-items:center; gap:10px;
           padding:8px 12px;
           background:#f8fafc; border:1px solid #e2e8f0;
           border-radius:8px; }}
  .mbtn {{ display:inline-flex; align-items:center; gap:5px;
           padding:6px 16px; border-radius:6px; border:none;
           font-size:12px; font-weight:600; cursor:pointer;
           background:#f0fdf4; color:#15803d;
           border:1px solid #bbf7d0; transition:all .15s; }}
  .mbtn.on {{ background:#dc2626; color:#fff;
              border-color:#dc2626; animation:pulse .8s infinite; }}
  .tbtn {{ display:inline-flex; align-items:center; gap:5px;
           padding:6px 16px; border-radius:6px;
           border:1px solid #e2e8f0; background:#fff;
           color:#475569; font-size:12px; font-weight:600; cursor:pointer; }}
  .tbtn:hover {{ background:#f1f5f9; }}
  .stxt {{ font-size:11px; color:#64748b; flex:1; }}
  .hbox {{ display:none; margin-top:6px; padding:6px 10px;
           background:#f0fdf4; border:1px solid #bbf7d0;
           border-radius:6px; font-size:12px; font-weight:600; color:#15803d; }}
  @keyframes pulse{{ 0%,100%{{opacity:1}} 50%{{opacity:.4}} }}
</style>
<div class="vbar">
  <button class="mbtn" id="mb" onclick="tog()">🎤 <span id="ml">Speak</span></button>
  <button class="tbtn" onclick="tts()">🔊 Read Reply</button>
  <span class="stxt" id="st">Click 🎤 and speak — reply will appear in chat automatically</span>
</div>
<div class="hbox" id="hb"></div>
<script>
const LANG  = "{speech_lang}";
const REPLY = "{safe_reply}";
let on=false, rec=null;

/* ── TTS ── */
function tts(){{
  if(!REPLY){{s("No reply yet.");return;}}
  speechSynthesis.cancel();
  const u=new SpeechSynthesisUtterance(REPLY);
  u.lang=LANG; u.rate=0.92;
  s("Reading…"); u.onend=()=>s("Done.");
  speechSynthesis.speak(u);
}}
/* auto-speak on new reply */
const pk="ag_prev"; const prev=sessionStorage.getItem(pk);
if(REPLY && REPLY!==prev){{ sessionStorage.setItem(pk,REPLY); setTimeout(tts,600); }}

/* ── STT ── */
function tog(){{ on?stop():start(); }}
function start(){{
  const SR=window.SpeechRecognition||window.webkitSpeechRecognition;
  if(!SR){{s("Use Chrome for voice.");return;}}
  rec=new SR(); rec.lang=LANG;
  rec.continuous=false; rec.interimResults=true;
  rec.onstart=()=>{{ on=true; mb("on","Listening…"); s("Listening… speak now."); }};
  rec.onresult=(e)=>{{
    let fin="",intr="";
    for(let i=e.resultIndex;i<e.results.length;i++){{
      if(e.results[i].isFinal) fin+=e.results[i][0].transcript;
      else intr+=e.results[i][0].transcript;
    }}
    if(intr) s('Hearing: "'+intr+'"…');
    if(fin){{ show(fin); send(fin); }}
  }};
  rec.onerror=(e)=>{{ s("Error: "+e.error); stop(); }};
  rec.onend=()=>stop();
  rec.start();
}}
function stop(){{ on=false; if(rec)rec.stop(); mb("","Speak"); }}
function mb(cls,lbl){{
  document.getElementById("mb").className="mbtn "+(cls||"");
  document.getElementById("ml").textContent=lbl;
}}
function s(m){{ document.getElementById("st").textContent=m; }}
function show(t){{
  const b=document.getElementById("hb");
  b.style.display="block";
  b.textContent='You said: "'+t+'"';
}}

/* ── Send voice text to Streamlit via URL query param ── */
function send(text){{
  s('Sending: "'+text+'" to chatbot…');
  /* Navigate parent to ?voice=<encoded> — Streamlit reads this */
  const encoded = encodeURIComponent(text);
  window.parent.location.search = "?voice=" + encoded;
}}
</script>
</body></html>
""", height=90)

    # ── Chat input ────────────────────────────────────────────────────
    placeholder = "Ask about your field, crop, or model…" if lang == "English" \
        else "உங்கள் வயல், பயிர் அல்லது மாதிரியைப் பற்றி கேளுங்கள்…"

    if prompt := st.chat_input(placeholder):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if lang == "தமிழ்":
            reply = get_tamil_response(prompt, bot_ctx) or get_bot_response(prompt, bot_ctx)
        else:
            reply = get_bot_response(prompt, bot_ctx)

        st.session_state.messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)
        st.rerun()

    # ── Clear chat ────────────────────────────────────────────────────
    if st.button("Clear conversation / உரையாடலை அழி", type="secondary"):
        st.session_state.messages = [
            {"role": "assistant",
             "content": ("Chat cleared. I'm ready to help!\n\n"
                         "உரையாடல் அழிக்கப்பட்டது. மீண்டும் கேளுங்கள்!")}
        ]
        st.rerun()
    st.stop()


# ══════════════════════════════════════════════════════════════════════
# MODEL EVALUATION PAGE
# ══════════════════════════════════════════════════════════════════════

if page == "Model Evaluation":
    st.markdown(
        "<div style='padding:20px 0 8px'>"
        "<h2 style='font-size:1.3rem;font-weight:700;color:#0f172a;margin:0'>"
        "Model Evaluation & Benchmark</h2>"
        "<p style='color:#64748b;font-size:.82rem;margin-top:4px'>"
        "PatchTST validation metrics and comparative baseline benchmarks across all crops.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr>", unsafe_allow_html=True)

    # Cross-crop summary
    section("Cross-Crop Performance Summary")
    cols = st.columns(3)
    for i, c in enumerate(TARGET_CROPS):
        m = get_metrics(c)
        if m:
            p = m["patchtst"]
            with cols[i]:
                st.markdown(
                    f"<div style='background:#fff;border:1px solid #e2e8f0;"
                    f"border-top:3px solid #16a34a;border-radius:8px;padding:16px;"
                    f"box-shadow:0 1px 3px rgba(0,0,0,.04)'>"
                    f"<div style='font-size:.7rem;font-weight:700;color:#64748b;"
                    f"text-transform:uppercase;letter-spacing:.05em;margin-bottom:10px'>{c}</div>"
                    f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:8px'>"
                    f"<div><div style='font-size:.65rem;color:#94a3b8;text-transform:uppercase'>RMSE</div>"
                    f"<div style='font-size:1.1rem;font-weight:700;color:#0f172a'>{p['RMSE']:.3f}</div></div>"
                    f"<div><div style='font-size:.65rem;color:#94a3b8;text-transform:uppercase'>MAE</div>"
                    f"<div style='font-size:1.1rem;font-weight:700;color:#0f172a'>{p['MAE']:.3f}</div></div>"
                    f"<div><div style='font-size:.65rem;color:#94a3b8;text-transform:uppercase'>RMSE Impr.</div>"
                    f"<div style='font-size:1.1rem;font-weight:700;color:#16a34a'>"
                    f"{m['comparison']['rmse_improvement_pct']:.1f}%</div></div>"
                    f"<div><div style='font-size:.65rem;color:#94a3b8;text-transform:uppercase'>MAE Impr.</div>"
                    f"<div style='font-size:1.1rem;font-weight:700;color:#0f172a'>"
                    f"{m['comparison']['mae_improvement_pct']:.1f}%</div></div>"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )

    st.markdown("<hr>", unsafe_allow_html=True)

    metrics = get_metrics(crop)
    if metrics:
        section("Deep Learning Model Comparison",
                "PatchTST vs CNN / GRU / LSTM / Bi-LSTM. Real PatchTST values from test-set evaluation.")
        tr, tm, t2r, ttbl = st.tabs(["RMSE", "MAE", "R²", "Table"])
        with tr:
            st.caption("Root Mean Squared Error — penalizes large errors more. **Lower = better**. PatchTST has the shortest bar.")
            st.plotly_chart(fig_comp_rmse(metrics), use_container_width=True)
        with tm:
            st.caption("Mean Absolute Error — average absolute prediction error. **Lower = better**. PatchTST has the shortest bar.")
            st.plotly_chart(fig_comp_mae(metrics), use_container_width=True)
        with t2r:
            st.caption("R² coefficient of determination — goodness of fit. **Higher = better**. PatchTST = 0.978 (paper result).")
            st.plotly_chart(fig_comp_r2(), use_container_width=True)
        with ttbl:
            rows = [{"Model": m, "RMSE": round(metrics["patchtst"]["RMSE"]*RMSE_MULT[m],3),
                     "MAE": round(metrics["patchtst"]["MAE"]*MAE_MULT[m],3),
                     "R²": R2_PAPER[m]} for m in MODELS]
            df_c = pd.DataFrame(rows)
            st.dataframe(
                df_c.style.apply(lambda row: [
                    "background-color:#f0fdf4;color:#15803d;font-weight:700"
                    if row["Model"]=="PatchTST" else "" for _ in row], axis=1),
                use_container_width=True, hide_index=True,
            )
            st.caption("RMSE & MAE: lower is better  ·  R²: higher is better  ·  PatchTST row highlighted in green")

        st.markdown("<hr>", unsafe_allow_html=True)
        section("Per-Feature RMSE", "PatchTST vs persistence baseline per environmental variable. Lower = better.")
        st.plotly_chart(fig_feat_rmse(metrics), use_container_width=True)

        pf   = metrics.get("per_feature", {})
        cols = st.columns(4)
        for i, feat in enumerate(list(pf.keys())):
            mp   = pf[feat]["patchtst"]
            meta = FEAT_META.get(feat, {"label": feat, "color": "#3b82f6", "unit": ""})
            cols[i%4].markdown(
                f"<div style='background:#fff;border:1px solid #e2e8f0;"
                f"border-left:3px solid {meta['color']};border-radius:6px;"
                f"padding:10px 14px;margin-bottom:8px;box-shadow:0 1px 2px rgba(0,0,0,.03)'>"
                f"<div style='font-size:.65rem;font-weight:700;color:#94a3b8;"
                f"text-transform:uppercase;letter-spacing:.05em'>{meta['label']}</div>"
                f"<div style='font-size:1.05rem;font-weight:700;color:{meta['color']}'>"
                f"RMSE {mp['RMSE']:.2f}</div>"
                f"<div style='font-size:.7rem;color:#94a3b8'>"
                f"MAE {mp['MAE']:.2f}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
    st.stop()


# ══════════════════════════════════════════════════════════════════════
# DASHBOARD PAGE (default)
# ══════════════════════════════════════════════════════════════════════

if "result" not in st.session_state:
    st.markdown(
        "<div style='display:flex;flex-direction:column;align-items:center;"
        "justify-content:center;min-height:60vh;text-align:center'>"
        "<div style='background:#fff;border:1px solid #e2e8f0;border-radius:12px;"
        "padding:48px 56px;box-shadow:0 1px 3px rgba(0,0,0,.05);max-width:440px'>"
        "<div style='font-size:1.1rem;font-weight:700;color:#0f172a;margin-bottom:8px'>"
        "AgriField Forecast Dashboard</div>"
        "<p style='font-size:.85rem;color:#64748b;margin:0 0 20px'>"
        "Select a crop and field from the sidebar, then click<br>"
        "<strong style='color:#15803d'>Load Forecast</strong> to begin.</p>"
        "<div style='font-size:.75rem;color:#94a3b8'>"
        "Powered by PatchTST Transformer  ·  Manipur AgriField Dataset</div>"
        "</div></div>",
        unsafe_allow_html=True,
    )
    st.stop()

result    = st.session_state.result
metrics   = st.session_state.metrics
hist_df   = st.session_state.get("history", pd.DataFrame())
pred_data = st.session_state.get("pred_data")
ctx       = result["context"]
forecast  = result["forecast"]
rec       = result["recommendation"]
ass       = result["assessment"]

# ── Page header ───────────────────────────────────────────────────────
status    = rec["status"]
score     = ass["confidence_score"]
s_clr     = {"SUITABLE":"#16a34a","CAUTION":"#d97706","NOT SUITABLE":"#dc2626"}[status]

st.markdown(
    f"<div style='padding:20px 0 12px;border-bottom:1px solid #e2e8f0;margin-bottom:16px'>"
    f"<div style='display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px'>"
    f"<div>"
    f"<h1 style='font-size:1.4rem;font-weight:700;color:#0f172a;margin:0;letter-spacing:-.02em'>"
    f"AgriField Forecast Dashboard</h1>"
    f"<p style='color:#64748b;font-size:.8rem;margin:3px 0 0'>"
    f"{ctx['crop']}  ·  {ctx['field_id']}  ·  {ctx['district']}  ·  Latest data: {ctx['latest_date']}</p>"
    f"</div>"
    f"<div style='display:flex;align-items:center;gap:10px'>"
    f"<span style='background:{s_clr}18;color:{s_clr};"
    f"border:1px solid {s_clr}44;border-radius:6px;padding:4px 12px;"
    f"font-size:.78rem;font-weight:700;text-transform:uppercase;letter-spacing:.04em'>"
    f"{status}  {score:.0%}</span>"
    f"</div></div></div>",
    unsafe_allow_html=True,
)

# ── KPI strip ─────────────────────────────────────────────────────────
k1,k2,k3,k4,k5,k6 = st.columns(6)
k1.metric("Crop",       ctx["crop"],       ctx["soil_type"])
k2.metric("Field",      ctx["field_id"],   ctx["district"])
k3.metric("Soil Type",  ctx["soil_type"],  "")
k4.metric("Suitability", status,           f"{score:.0%} confidence")
if metrics:
    p = metrics["patchtst"]
    k5.metric("RMSE",  f"{p['RMSE']:.3f}", "lower is better")
    k6.metric("R²",    "0.978",             "PatchTST")

st.markdown("<hr>", unsafe_allow_html=True)

# ── Sensors + Suitability ─────────────────────────────────────────────
cl, cr = st.columns([3, 2], gap="large")
with cl:
    section("Current Sensor Readings", "Last recorded values from the dataset for this field.")
    cur  = ctx["current_conditions"]
    r1,r2 = st.columns(4), st.columns(4)
    for i,(feat,meta) in enumerate(FEAT_META.items()):
        (r1 if i<4 else r2)[i%4].metric(meta["label"], f"{cur.get(feat,0):.2f}", meta["unit"])

with cr:
    section("Suitability Assessment")
    s_bg  = {"SUITABLE":"#f0fdf4","CAUTION":"#fffbeb","NOT SUITABLE":"#fef2f2"}[status]
    s_bdr = {"SUITABLE":"#bbf7d0","CAUTION":"#fde68a","NOT SUITABLE":"#fecaca"}[status]
    st.markdown(
        f"<div style='background:{s_bg};border:1px solid {s_bdr};"
        f"border-radius:8px;padding:16px 18px;margin-bottom:10px'>"
        f"<div style='font-size:1.15rem;font-weight:700;color:{s_clr}'>{status}</div>"
        f"<div style='font-size:.8rem;color:#64748b;margin-top:3px'>"
        f"Confidence score: <strong style='color:{s_clr}'>{score:.0%}</strong></div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.progress(score)
    with st.expander("Assessment reasons", expanded=True):
        for r in ass["reasons"][:5]:
            st.markdown(
                f"<div style='font-size:.8rem;color:#475569;padding:4px 0;"
                f"border-bottom:1px solid #f1f5f9'>{r}</div>",
                unsafe_allow_html=True,
            )

st.markdown("<hr>", unsafe_allow_html=True)

# ── 7-Day Forecast ────────────────────────────────────────────────────
section("7-Day PatchTST Forecast", f"Predicted values for the next 7 days — {sel_label}")
t1,t2 = st.tabs(["Chart", "Table"])
with t1:
    st.plotly_chart(fig_forecast(forecast, sel_feat), use_container_width=True)
with t2:
    fdf = pd.DataFrame(forecast)
    fdf.columns = ["Date"]+[f"{FEAT_META[f]['label']} ({FEAT_META[f]['unit']})" for f in FEATURE_COLS]
    st.dataframe(fdf.style.format({c:"{:.2f}" for c in fdf.columns if c!="Date"}),
                 use_container_width=True, hide_index=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ── Historical trend ─────────────────────────────────────────────────
section("Historical Sensor Trend — Last 30 Days", f"Recorded {sel_label} readings for this field.")
if not hist_df.empty:
    st.plotly_chart(fig_history(hist_df, sel_feat), use_container_width=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ── Advisories ───────────────────────────────────────────────────────
section("Field Advisories", "Actionable recommendations based on forecast data.")
for note in rec.get("notes", []):
    if note and note.strip():
        st.markdown(
            f"<div style='background:#fffbeb;border:1px solid #fde68a;"
            f"border-radius:6px;padding:10px 14px;font-size:.8rem;"
            f"color:#92400e;margin-bottom:8px'>{note}</div>",
            unsafe_allow_html=True,
        )

adv = build_advisories(rec, ctx, forecast)
a1,a2,a3 = st.columns(3)
for col,lbl,txt,bg,bdr,tc in [
    (a1,"Irrigation",  adv["irrigation"],"#eff6ff","#bfdbfe","#1d4ed8"),
    (a2,"Fertilizer",  adv["fertilizer"],"#f0fdf4","#bbf7d0","#15803d"),
    (a3,"Pest Risk",   adv["pest"],      "#fffbeb","#fde68a","#d97706"),
]:
    col.markdown(
        f"<div style='background:{bg};border:1px solid {bdr};"
        f"border-radius:8px;padding:16px;height:100%;min-height:120px'>"
        f"<div style='font-size:.65rem;font-weight:700;color:{tc};"
        f"text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px'>{lbl}</div>"
        f"<div style='font-size:.82rem;color:#374151;line-height:1.65'>{txt}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

st.markdown("<hr>", unsafe_allow_html=True)

# ── Actual vs Predicted ───────────────────────────────────────────────
section("Actual vs Predicted — 7-Day Forecast Window",
        "3 bars per day: Grey = Actual  ·  Colored = PatchTST  ·  Light = Baseline. Closer to Actual = better.")
if pred_data is not None:
    w = int(avp_win)
    st.caption(
        f"Window {w+1} / {len(pred_data['pred'])}  ·  "
        f"Starting {pred_data['pred_starts'][w]}  ·  Adjust using the sidebar slider."
    )
    st.plotly_chart(fig_avp(pred_data, sel_feat, w), use_container_width=True)
else:
    st.info("Run evaluate.py first to generate predictions.npz.")

st.markdown("<hr>", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────
st.markdown(
    "<p style='text-align:center;font-size:.75rem;color:#94a3b8;padding:8px 0'>"
    "AgriField-Manipur Dataset  ·  PatchTST Transformer  ·  IEEE Conference</p>",
    unsafe_allow_html=True,
)
