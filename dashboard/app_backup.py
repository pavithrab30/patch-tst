"""Streamlit dashboard for AgriField PatchTST forecasting."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agri_system.config import TARGET_CROPS
from agri_system.inference import forecast_field, list_fields, load_metrics

st.set_page_config(page_title="AgriField PatchTST Dashboard", layout="wide")
st.title("AgriField")

with st.expander("What does this dashboard show?", expanded=False):
    st.markdown("""
**In short:** Pick a crop and field. The dashboard shows **today's conditions**, **next 7 days predicted by PatchTST**,
and whether those predicted conditions look **good, mixed, or poor** for that crop — based on patterns in your dataset.

| Section | Meaning |
|---------|---------|
| **Field Overview** | Basic field info and whether the dataset labeled this field as suitable (`Yes`/`No`) on the most recent day. |
| **Current Conditions** | Last recorded readings from the CSV: temperature, humidity, soil moisture, pH, N/P/K, rainfall. |
| **7-Day Forecast** | PatchTST prediction for the next 7 days using the past 30 days of that field's data. |
| **Suitability Assessment** | Compares the **average of the 7-day forecast** to historical ranges where the crop was labeled suitable vs not suitable in training data. **SUITABLE** = forecast looks like good days; **CAUTION** = mixed; **NOT SUITABLE** = forecast looks like bad days. |
| **Advisories** | Irrigation, fertilizer, and pest notes taken directly from the dataset for that field (not invented). |
| **Model Metrics** *(optional)* | Overall error scores on test data — mainly for checking how reliable the model is. |
    """)

col1, col2 = st.columns(2)
with col1:
    crop = st.selectbox("Crop", TARGET_CROPS)
with col2:
    fields = list_fields(crop)
    field_id = st.selectbox("Field", fields)

if st.button("Load forecast", type="primary") or "loaded" not in st.session_state:
    try:
        result = forecast_field(crop, field_id)
        if not result:
            # forecast_field ran without raising but gave back nothing usable
            # (None / {} / empty). Surface that clearly instead of silently
            # falling through to the "select crop and field" screen.
            st.error(
                f"forecast_field(crop={crop!r}, field_id={field_id!r}) returned "
                f"an empty result ({result!r}). Check that this field has enough "
                "history (30+ days) and that the crop/field combination is valid."
            )
            st.stop()
        st.session_state.metrics = load_metrics(crop)
        st.session_state.result = result
        st.session_state.loaded = True  # only set once result+metrics succeeded
    except Exception as e:
        st.error(f"Error loading forecast: {e}")
        st.exception(e)  # full traceback, remove once debugged
        st.stop()

result = st.session_state.get("result")
if not result:
    st.info("Select crop and field, then click Load forecast.")
    st.stop()

ctx = result["context"]
forecast = result["forecast"]
rec = result["recommendation"]
metrics = st.session_state.get("metrics", {})

st.subheader("Field Overview")
c1, c2 = st.columns(2)
c1.metric("Soil type", ctx["soil_type"])
c2.metric(
    "Dataset suitability (latest record)",
    ctx["crop_suitable"],
    help="Yes/No label from the CSV for the most recent day — not a live sensor reading.",
)

st.subheader("Current Soil & Weather Conditions")
st.caption("Last recorded values from the dataset for this field.")
cur_df = pd.DataFrame([ctx["current_conditions"]]).T
cur_df.columns = ["Value"]
st.dataframe(cur_df, width="stretch")

st.subheader("PatchTST 7-Day Forecast")
st.caption("Predicted values for the next 7 days, based on the last 30 days of this field's history.")
st.dataframe(forecast, width="stretch")

status_color = {"SUITABLE": "green", "CAUTION": "orange", "NOT SUITABLE": "red"}
st.subheader("Field Suitability Assessment")
st.caption(
    "Based on whether the 7-day forecast average matches conditions seen when this crop was "
    "labeled suitable (Yes) or not suitable (No) in training data."
)
st.markdown(
    f"**Status:** :{status_color.get(rec['status'], 'gray')}[{rec['status']}]  "
    f"(score {rec['confidence_score']:.0%} — higher means forecast looks more like suitable conditions)"
)
with st.expander("Reasons"):
    for r in rec["reasons"]:
        st.write(f"- {r}")

st.subheader("Advisories")
st.caption("Taken from the dataset record for this field — irrigation/fertilizer/pest guidance already in the CSV.")
a1, a2, a3 = st.columns(3)
a1.info(f"**Irrigation:** {rec['irrigation_advice']}")
a2.info(f"**Fertilizer:** {rec['fertilizer_advice']}")
a3.warning(f"**Pest risk:** {rec['pest_risk']}")
if rec["notes"]:
    for n in rec["notes"]:
        st.warning(n)

with st.expander("Model accuracy (optional — for technical review)"):
    if metrics:
        m1, m2, m3 = st.columns(3)
        p = metrics["patchtst"]
        b = metrics["baseline"]
        m1.metric("PatchTST RMSE", f"{p['RMSE']:.4f}")
        m2.metric("PatchTST R²", f"{p['R2']:.4f}")
        m3.metric("PatchTST MAE", f"{p['MAE']:.4f}")

        comp_df = pd.DataFrame(
            {
                "Metric": ["MAE", "RMSE", "R²", "MAPE"],
                "PatchTST": [p["MAE"], p["RMSE"], p["R2"], p["MAPE"]],
                "Baseline": [b["MAE"], b["RMSE"], b["R2"], b["MAPE"]],
            }
        )
        st.dataframe(comp_df, width="stretch")
        st.caption(
            f"Train loss={metrics.get('train_loss', 'N/A'):.6f} | "
            f"Val loss={metrics.get('validation_loss', 'N/A'):.6f} | "
            f"RMSE improvement vs baseline: {metrics['comparison']['rmse_improvement_pct']:.1f}%"
        )
    else:
        st.info("Metrics not found. Run training and evaluation first.")