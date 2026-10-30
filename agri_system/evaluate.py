"""Phase 3: Evaluate PatchTST and persistence baseline."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from agri_system.config import FEATURE_COLS, MODELS_DIR, OUTPUTS_DIR, PRED_LEN, TARGET_CROPS
from agri_system.dataset import build_dataloaders
from agri_system.metrics import compute_all_metrics, persistence_baseline
from agri_system.model_utils import build_model, get_device


def load_trained_model(crop: str):
    crop_dir = MODELS_DIR / crop.lower()
    checkpoint = torch.load(crop_dir / "checkpoint.pth", map_location="cpu", weights_only=False)
    model, args = build_model(use_gpu=torch.cuda.is_available())
    model.load_state_dict(checkpoint["model_state_dict"])
    device = get_device(args)
    model = model.to(device)
    model.eval()

    with open(crop_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, args, scaler, device


def _collect_predictions(model, loader, device, scaler):
    preds, trues, inputs, field_ids, pred_starts = [], [], [], [], []

    with torch.no_grad():
        for batch_x, batch_y, fields, starts in loader:
            batch_x_dev = batch_x.to(device)
            outputs = model(batch_x_dev)
            outputs = outputs[:, -PRED_LEN:, :].cpu().numpy()
            target = batch_y[:, -PRED_LEN:, :].numpy()
            inp = batch_x.numpy()

            preds.append(outputs)
            trues.append(target)
            inputs.append(inp)
            field_ids.extend(fields)
            pred_starts.extend(starts)

    pred = np.concatenate(preds, axis=0)
    true = np.concatenate(trues, axis=0)
    inp = np.concatenate(inputs, axis=0)

    pred_inv = scaler.inverse_transform(pred.reshape(-1, len(FEATURE_COLS))).reshape(pred.shape)
    true_inv = scaler.inverse_transform(true.reshape(-1, len(FEATURE_COLS))).reshape(true.shape)
    inp_inv = scaler.inverse_transform(inp.reshape(-1, len(FEATURE_COLS))).reshape(inp.shape)

    baseline = persistence_baseline(inp, PRED_LEN)
    baseline_inv = scaler.inverse_transform(baseline.reshape(-1, len(FEATURE_COLS))).reshape(baseline.shape)

    return {
        "pred": pred_inv,
        "true": true_inv,
        "baseline": baseline_inv,
        "field_ids": field_ids,
        "pred_starts": pred_starts,
    }


def evaluate_crop(crop: str) -> dict:
    print(f"\nEvaluating {crop}...")
    loaders, _, metadata = build_dataloaders(crop, batch_size=32, num_workers=0)
    model, args, scaler, device = load_trained_model(crop)

    collected = _collect_predictions(model, loaders["test"], device, scaler)

    patchtst_metrics = compute_all_metrics(collected["pred"], collected["true"])
    baseline_metrics = compute_all_metrics(collected["baseline"], collected["true"])

    per_feature = {}
    for i, feat in enumerate(FEATURE_COLS):
        per_feature[feat] = {
            "patchtst": compute_all_metrics(collected["pred"][:, :, i], collected["true"][:, :, i]),
            "baseline": compute_all_metrics(collected["baseline"][:, :, i], collected["true"][:, :, i]),
        }

    with open(MODELS_DIR / crop.lower() / "config.json", encoding="utf-8") as f:
        train_config = json.load(f)

    results = {
        "crop": crop,
        "test_samples": len(collected["pred"]),
        "train_loss": train_config.get("final_train_loss"),
        "validation_loss": train_config.get("best_val_loss"),
        "patchtst": patchtst_metrics,
        "baseline": baseline_metrics,
        "per_feature": per_feature,
        "comparison": {
            "rmse_improvement_pct": (
                (baseline_metrics["RMSE"] - patchtst_metrics["RMSE"]) / baseline_metrics["RMSE"] * 100
                if baseline_metrics["RMSE"] > 0
                else 0.0
            ),
            "mae_improvement_pct": (
                (baseline_metrics["MAE"] - patchtst_metrics["MAE"]) / baseline_metrics["MAE"] * 100
                if baseline_metrics["MAE"] > 0
                else 0.0
            ),
        },
    }

    out_dir = OUTPUTS_DIR / crop.lower()
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    np.savez_compressed(
        out_dir / "predictions.npz",
        pred=collected["pred"],
        true=collected["true"],
        baseline=collected["baseline"],
        field_ids=np.array(collected["field_ids"]),
        pred_starts=np.array(collected["pred_starts"]),
        feature_cols=np.array(FEATURE_COLS),
    )

    _plot_actual_vs_predicted(collected, out_dir / "actual_vs_predicted.png")

    print(f"  PatchTST RMSE={patchtst_metrics['RMSE']:.4f} R2={patchtst_metrics['R2']:.4f}")
    print(f"  Baseline RMSE={baseline_metrics['RMSE']:.4f} R2={baseline_metrics['R2']:.4f}")
    return results


def _plot_actual_vs_predicted(collected: dict, path: Path, feature_idx: int = 2):
    """Plot soil_moisture (default index 2) for first test window."""
    feat_name = FEATURE_COLS[feature_idx]
    true = collected["true"][0, :, feature_idx]
    pred = collected["pred"][0, :, feature_idx]
    base = collected["baseline"][0, :, feature_idx]

    plt.figure(figsize=(9, 4))
    days = np.arange(1, PRED_LEN + 1)
    plt.plot(days, true, "o-", label="Actual", linewidth=2)
    plt.plot(days, pred, "s--", label="PatchTST", linewidth=2)
    plt.plot(days, base, "^:", label="Persistence baseline", linewidth=2)
    plt.xlabel("Forecast day")
    plt.ylabel(feat_name)
    plt.title(f"Actual vs Predicted — {feat_name} (first test window)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def evaluate_all_crops() -> dict:
    all_results = {}
    for crop in TARGET_CROPS:
        all_results[crop] = evaluate_crop(crop)

    summary_path = OUTPUTS_DIR / "evaluation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    return all_results


if __name__ == "__main__":
    evaluate_all_crops()
