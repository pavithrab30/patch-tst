"""Phase 4: Decision engine using real forecasts and data-derived thresholds."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from agri_system.config import FEATURE_COLS, MODELS_DIR, PRED_LEN, PROCESSED_DIR


def load_thresholds(crop: str) -> dict:
    meta_path = PROCESSED_DIR / crop.lower() / "metadata.json"
    with open(meta_path, encoding="utf-8") as f:
        metadata = json.load(f)
    return metadata["thresholds"], metadata


def _score_feature(value: float, thr: dict) -> float:
    """Higher score = closer to suitable (Yes) distribution."""
    if thr["no_p50"] is None:
        dist_to_yes = abs(value - thr["yes_p50"])
        spread = max(thr["yes_p75"] - thr["yes_p25"], 1e-6)
        return max(0.0, 1.0 - dist_to_yes / (2 * spread))

    yes_center = thr["yes_p50"]
    no_center = thr["no_p50"]
    midpoint = (yes_center + no_center) / 2
    if value >= yes_center and yes_center >= no_center:
        return 1.0
    if value <= no_center and no_center <= yes_center:
        return 0.0
    if yes_center > no_center:
        if value >= yes_center:
            return 1.0
        if value <= no_center:
            return 0.0
        return (value - no_center) / max(yes_center - no_center, 1e-6)
    else:
        if value <= yes_center:
            return 1.0
        if value >= no_center:
            return 0.0
        return (no_center - value) / max(no_center - yes_center, 1e-6)


def assess_forecast(forecast: np.ndarray, thresholds: dict) -> dict:
    """
    forecast: shape (pred_len, n_features) in original units.
    Returns suitability status and per-feature reasons.
    """
    feature_scores = {}
    reasons = []

    for i, feat in enumerate(FEATURE_COLS):
        if feat not in thresholds:
            continue
        vals = forecast[:, i]
        mean_val = float(np.mean(vals))
        score = _score_feature(mean_val, thresholds[feat])
        feature_scores[feat] = {"mean": mean_val, "score": score}

        thr = thresholds[feat]
        if score >= 0.65:
            reasons.append(f"{feat}={mean_val:.2f} within suitable range (Yes median={thr['yes_p50']:.2f})")
        elif score >= 0.35:
            reasons.append(f"{feat}={mean_val:.2f} is borderline (Yes={thr['yes_p50']:.2f}, No={thr['no_p50']})")
        else:
            reasons.append(f"{feat}={mean_val:.2f} aligns with unsuitable conditions (No median={thr['no_p50']})")

    avg_score = float(np.mean([v["score"] for v in feature_scores.values()]))

    if avg_score >= 0.65:
        status = "SUITABLE"
    elif avg_score >= 0.40:
        status = "CAUTION"
    else:
        status = "NOT SUITABLE"

    return {
        "status": status,
        "confidence_score": avg_score,
        "feature_scores": feature_scores,
        "reasons": reasons,
    }


def get_field_context(crop: str, field_id: str) -> dict:
    df = pd.read_csv(
        PROCESSED_DIR / crop.lower() / f"{crop.lower()}_processed.csv",
        parse_dates=["date"],
    )
    field_df = df[df["field_id"] == field_id].sort_values("date")
    if field_df.empty:
        raise ValueError(f"Field {field_id} not found for crop {crop}")

    latest = field_df.iloc[-1]
    return {
        "field_id": field_id,
        "crop": crop,
        "soil_type": latest["soil_type"],
        "district": latest["district"],
        "latest_date": str(latest["date"].date()),
        "current_conditions": {col: float(latest[col]) for col in FEATURE_COLS},
        "suitability_score": float(latest["suitability_score"]),
        "crop_suitable": latest["crop_suitable"],
        "irrigation_advice": latest["irrigation_advice"],
        "fertilizer_advice": latest["fertilizer_advice"],
        "pest_risk": latest["pest_risk"],
    }


def build_recommendation(context: dict, assessment: dict, forecast_df: pd.DataFrame) -> dict:
    rec = {
        "status": assessment["status"],
        "confidence_score": assessment["confidence_score"],
        "reasons": assessment["reasons"][:5],
        "irrigation_advice": context["irrigation_advice"],
        "fertilizer_advice": context["fertilizer_advice"],
        "pest_risk": context["pest_risk"],
        "notes": [],
    }

    avg_soil = float(forecast_df["soil_moisture"].mean())
    avg_rain = float(forecast_df["rainfall"].mean())

    if avg_soil < forecast_df["soil_moisture"].median() * 0.85:
        rec["notes"].append(
            f"Forecast soil moisture ({avg_soil:.1f}%) trending low — monitor irrigation needs."
        )
    if avg_rain > forecast_df["rainfall"].quantile(0.75):
        rec["notes"].append(
            f"Elevated rainfall expected ({avg_rain:.1f} mm avg) — check drainage and pest risk."
        )
    if assessment["status"] == "NOT SUITABLE":
        rec["notes"].append(
            "Field suitability for this crop is unfavorable based on forecast vs training-derived thresholds."
        )
    elif assessment["status"] == "CAUTION":
        rec["notes"].append(
            "Mixed signals — some forecast variables align with unsuitable historical conditions."
        )

    return rec
