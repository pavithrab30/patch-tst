"""Metrics utilities."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import r2_score


def mae(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - true)))


def mse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean((pred - true) ** 2))


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(mse(pred, true)))


def mape(pred: np.ndarray, true: np.ndarray) -> float:
    denom = np.where(np.abs(true) < 1e-8, 1e-8, true)
    return float(np.mean(np.abs((pred - true) / denom)))


def r2(pred: np.ndarray, true: np.ndarray) -> float:
    return float(r2_score(true.reshape(-1), pred.reshape(-1)))


def compute_all_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    return {
        "MAE": mae(pred, true),
        "MSE": mse(pred, true),
        "RMSE": rmse(pred, true),
        "MAPE": mape(pred, true),
        "R2": r2(pred, true),
    }


def persistence_baseline(batch_x: np.ndarray, pred_len: int) -> np.ndarray:
    """Repeat last observed timestep for each forecast horizon."""
    last = batch_x[:, -1:, :]
    return np.repeat(last, pred_len, axis=1)
