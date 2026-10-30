"""Phase 1: Data inspection and preprocessing for AgriField-Manipur."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from agri_system.config import (
    DATASET_PATH,
    FEATURE_COLS,
    PROCESSED_DIR,
    TARGET_CROPS,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)


def find_dataset() -> Path:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
    return DATASET_PATH


def inspect_dataset(df: pd.DataFrame) -> dict:
    inspection = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": df.isnull().sum().astype(int).to_dict(),
        "duplicate_field_date": int(df.duplicated(["field_id", "date"]).sum()),
        "date_range": {"min": str(df["date"].min()), "max": str(df["date"].max())},
        "crops": df["crop"].value_counts().astype(int).to_dict(),
        "soil_types": df["soil_type"].value_counts().astype(int).to_dict(),
        "field_id_count": int(df["field_id"].nunique()),
        "crop_suitable": df["crop_suitable"].value_counts().astype(int).to_dict(),
    }
    return inspection


def _chronological_split_dates(dates: pd.Series) -> dict[str, tuple[str, str]]:
    unique_dates = sorted(pd.to_datetime(dates).unique())
    n = len(unique_dates)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)

    return {
        "train": (str(unique_dates[0].date()), str(unique_dates[train_end - 1].date())),
        "val": (str(unique_dates[train_end].date()), str(unique_dates[val_end - 1].date())),
        "test": (str(unique_dates[val_end].date()), str(unique_dates[-1].date())),
        "train_end_idx": train_end,
        "val_end_idx": val_end,
        "total_days": n,
    }


def process_crop(df: pd.DataFrame, crop: str) -> tuple[pd.DataFrame, dict]:
    crop_df = df[df["crop"] == crop].copy()
    if crop_df.empty:
        raise ValueError(f"No records found for crop: {crop}")

    crop_df["date"] = pd.to_datetime(crop_df["date"])
    crop_df = crop_df.sort_values(["field_id", "date"]).reset_index(drop=True)

    split_info = _chronological_split_dates(crop_df["date"])
    fields = sorted(crop_df["field_id"].unique())

    per_field = {}
    for field_id in fields:
        field_df = crop_df[crop_df["field_id"] == field_id]
        per_field[field_id] = {
            "rows": int(len(field_df)),
            "date_min": str(field_df["date"].min().date()),
            "date_max": str(field_df["date"].max().date()),
            "soil_type": field_df["soil_type"].iloc[0],
            "district": field_df["district"].iloc[0],
        }

    processed = crop_df[
        ["date", "field_id", "district", "soil_type", "crop"]
        + FEATURE_COLS
        + [
            "suitability_score",
            "crop_suitable",
            "irrigation_advice",
            "fertilizer_advice",
            "pest_risk",
        ]
    ].copy()

    metadata = {
        "crop": crop,
        "rows": int(len(processed)),
        "fields": fields,
        "per_field": per_field,
        "split": split_info,
        "feature_columns": FEATURE_COLS,
        "missing_after_filter": processed[FEATURE_COLS].isnull().sum().astype(int).to_dict(),
    }
    return processed, metadata


def derive_thresholds(train_df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Derive suitability thresholds from training split only."""
    yes = train_df[train_df["crop_suitable"] == "Yes"]
    no = train_df[train_df["crop_suitable"] == "No"]

    thresholds = {}
    for col in feature_cols:
        yes_vals = yes[col].dropna()
        no_vals = no[col].dropna()
        thresholds[col] = {
            "yes_p25": float(yes_vals.quantile(0.25)),
            "yes_p50": float(yes_vals.quantile(0.50)),
            "yes_p75": float(yes_vals.quantile(0.75)),
            "no_p25": float(no_vals.quantile(0.25)) if len(no_vals) else None,
            "no_p50": float(no_vals.quantile(0.50)) if len(no_vals) else None,
            "no_p75": float(no_vals.quantile(0.75)) if len(no_vals) else None,
            "overall_median": float(train_df[col].median()),
        }
    return thresholds


def run_preprocessing() -> dict:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    dataset_path = find_dataset()
    df = pd.read_csv(dataset_path)

    report = {"dataset_path": str(dataset_path), "inspection": inspect_dataset(df), "crops": {}}

    for crop in TARGET_CROPS:
        processed, metadata = process_crop(df, crop)
        crop_dir = PROCESSED_DIR / crop.lower()
        crop_dir.mkdir(parents=True, exist_ok=True)

        out_csv = crop_dir / f"{crop.lower()}_processed.csv"
        processed.to_csv(out_csv, index=False)

        split = metadata["split"]
        train_end = split["train_end_idx"]
        val_end = split["val_end_idx"]
        unique_dates = sorted(processed["date"].unique())
        train_cutoff = unique_dates[train_end - 1]
        val_cutoff = unique_dates[val_end - 1]

        train_mask = processed["date"] <= train_cutoff
        thresholds = derive_thresholds(processed[train_mask], FEATURE_COLS)

        metadata["thresholds"] = thresholds
        metadata["processed_csv"] = str(out_csv)

        meta_path = crop_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        report["crops"][crop] = {
            "processed_csv": str(out_csv),
            "metadata": str(meta_path),
            "rows": metadata["rows"],
            "fields": metadata["fields"],
            "split": split,
        }

    report_path = PROCESSED_DIR / "data_inspection.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Preprocessing complete. Report saved to {report_path}")
    return report


if __name__ == "__main__":
    run_preprocessing()
