"""PyTorch dataset for per-field agricultural time series."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from agri_system.config import FEATURE_COLS, LABEL_LEN, PRED_LEN, PROCESSED_DIR, SEQ_LEN


class AgriTimeSeriesDataset(Dataset):
    """Sliding-window dataset preserving separate field_id sequences."""

    def __init__(
        self,
        df: pd.DataFrame,
        field_ids: list[str],
        date_start: pd.Timestamp,
        date_end: pd.Timestamp,
        pred_start: pd.Timestamp,
        pred_end: pd.Timestamp,
        scaler: StandardScaler | None = None,
        fit_scaler: bool = False,
    ):
        self.feature_cols = FEATURE_COLS
        self.seq_len = SEQ_LEN
        self.label_len = LABEL_LEN
        self.pred_len = PRED_LEN
        self.samples: list[tuple[np.ndarray, np.ndarray, str, pd.Timestamp]] = []

        if fit_scaler:
            self.scaler = StandardScaler()
            fit_rows = df[(df["date"] >= date_start) & (df["date"] <= date_end)]
            self.scaler.fit(fit_rows[self.feature_cols].values)
        else:
            if scaler is None:
                raise ValueError("Scaler required when fit_scaler=False")
            self.scaler = scaler

        for field_id in field_ids:
            field_df = df[df["field_id"] == field_id].sort_values("date").reset_index(drop=True)
            values = self.scaler.transform(field_df[self.feature_cols].values)
            dates = field_df["date"].tolist()

            for i in range(len(field_df) - self.seq_len - self.pred_len + 1):
                x_dates = dates[i : i + self.seq_len]
                y_dates = dates[i + self.seq_len - self.label_len : i + self.seq_len + self.pred_len]

                pred_dates = y_dates[-self.pred_len :]
                if not (pred_dates[0] >= pred_start and pred_dates[-1] <= pred_end):
                    continue
                if x_dates[-1] > date_end:
                    continue

                seq_x = values[i : i + self.seq_len]
                r_begin = i + self.seq_len - self.label_len
                r_end = r_begin + self.label_len + self.pred_len
                seq_y = values[r_begin:r_end]
                self.samples.append((seq_x, seq_y, field_id, pred_dates[0]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        seq_x, seq_y, field_id, pred_start = self.samples[index]
        return (
            torch.tensor(seq_x, dtype=torch.float32),
            torch.tensor(seq_y, dtype=torch.float32),
            field_id,
            str(pred_start.date()),
        )


def load_crop_data(crop: str) -> tuple[pd.DataFrame, dict]:
    crop_dir = PROCESSED_DIR / crop.lower()
    csv_path = crop_dir / f"{crop.lower()}_processed.csv"
    meta_path = crop_dir / "metadata.json"

    df = pd.read_csv(csv_path, parse_dates=["date"])
    with open(meta_path, encoding="utf-8") as f:
        metadata = json.load(f)
    return df, metadata


def get_split_dates(metadata: dict) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    df_dates = pd.to_datetime(
        [
            metadata["split"]["train"][0],
            metadata["split"]["train"][1],
            metadata["split"]["val"][0],
            metadata["split"]["val"][1],
            metadata["split"]["test"][0],
            metadata["split"]["test"][1],
        ]
    )
    return {
        "train": (df_dates[0], df_dates[1]),
        "val": (df_dates[2], df_dates[3]),
        "test": (df_dates[4], df_dates[5]),
    }


def build_dataloaders(crop: str, batch_size: int, num_workers: int = 0):
    df, metadata = load_crop_data(crop)
    splits = get_split_dates(metadata)
    fields = metadata["fields"]

    train_ds = AgriTimeSeriesDataset(
        df,
        fields,
        splits["train"][0],
        splits["train"][1],
        splits["train"][0],
        splits["train"][1],
        fit_scaler=True,
    )
    scaler = train_ds.scaler

    val_ds = AgriTimeSeriesDataset(
        df,
        fields,
        splits["train"][0],
        splits["val"][1],
        splits["val"][0],
        splits["val"][1],
        scaler=scaler,
    )
    test_ds = AgriTimeSeriesDataset(
        df,
        fields,
        splits["train"][0],
        splits["test"][1],
        splits["test"][0],
        splits["test"][1],
        scaler=scaler,
    )

    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }
    return loaders, scaler, metadata
