"""Shared inference utilities for console and dashboard."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from agri_system.config import FEATURE_COLS, MODELS_DIR, PRED_LEN, PROCESSED_DIR, SEQ_LEN
from agri_system.decision_engine import assess_forecast, build_recommendation, get_field_context, load_thresholds
from agri_system.evaluate import load_trained_model


def list_fields(crop: str) -> list[str]:
    meta_path = PROCESSED_DIR / crop.lower() / "metadata.json"
    with open(meta_path, encoding="utf-8") as f:
        metadata = json.load(f)
    return metadata["fields"]


def load_metrics(crop: str) -> dict:
    metrics_path = Path(__file__).resolve().parent.parent / "artifacts" / "outputs" / crop.lower() / "metrics.json"
    if not metrics_path.exists():
        return {}
    with open(metrics_path, encoding="utf-8") as f:
        return json.load(f)


def forecast_field(crop: str, field_id: str) -> dict:
    context = get_field_context(crop, field_id)
    thresholds, metadata = load_thresholds(crop)
    model, args, scaler, device = load_trained_model(crop)

    df = pd.read_csv(
        PROCESSED_DIR / crop.lower() / f"{crop.lower()}_processed.csv",
        parse_dates=["date"],
    )
    field_df = df[df["field_id"] == field_id].sort_values("date")
    if len(field_df) < SEQ_LEN:
        raise ValueError(f"Field {field_id} has insufficient history ({len(field_df)} days)")

    recent = field_df.tail(SEQ_LEN)
    x = scaler.transform(recent[FEATURE_COLS].values)
    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x_tensor)
        pred_scaled = out[:, -PRED_LEN:, :].cpu().numpy()[0]

    pred = scaler.inverse_transform(pred_scaled)
    forecast_dates = pd.date_range(
        start=recent["date"].iloc[-1] + pd.Timedelta(days=1),
        periods=PRED_LEN,
        freq="D",
    )

    forecast_df = pd.DataFrame(pred, columns=FEATURE_COLS)
    forecast_df.insert(0, "date", forecast_dates)

    assessment = assess_forecast(pred, thresholds)
    recommendation = build_recommendation(context, assessment, forecast_df)

    return {
        "context": context,
        "forecast": forecast_df,
        "assessment": assessment,
        "recommendation": recommendation,
        "metadata": metadata,
    }


def load_evaluation_predictions(crop: str) -> dict | None:
    npz_path = Path(__file__).resolve().parent.parent / "artifacts" / "outputs" / crop.lower() / "predictions.npz"
    if not npz_path.exists():
        return None
    data = np.load(npz_path, allow_pickle=True)
    return {
        "pred": data["pred"],
        "true": data["true"],
        "baseline": data["baseline"],
        "field_ids": data["field_ids"],
        "pred_starts": data["pred_starts"],
        "feature_cols": list(data["feature_cols"]),
    }
