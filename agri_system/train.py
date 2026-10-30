"""Phase 2: Train PatchTST models for each target crop."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler

from agri_system.config import (
    LABEL_LEN,
    MODELS_DIR,
    OUTPUTS_DIR,
    PRED_LEN,
    TARGET_CROPS,
    TRAIN_HPARAMS,
)
from agri_system.dataset import build_dataloaders
from agri_system.model_utils import build_model, get_device


class EarlyStopping:
    def __init__(self, patience: int = 5):
        self.patience = patience
        self.counter = 0
        self.best_loss = float("inf")
        self.best_state = None
        self.early_stop = False

    def __call__(self, val_loss: float, model: nn.Module):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def _run_epoch(model, loader, criterion, optimizer, device, train: bool):
    losses = []
    if train:
        model.train()
    else:
        model.eval()

    for batch_x, batch_y, _, _ in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            outputs = model(batch_x)
            outputs = outputs[:, -PRED_LEN:, :]
            target = batch_y[:, -PRED_LEN:, :]
            loss = criterion(outputs, target)

            if train:
                loss.backward()
                optimizer.step()

        losses.append(loss.item())
    return float(np.mean(losses))


def train_crop(crop: str) -> dict:
    print(f"\n{'=' * 60}\nTraining PatchTST for {crop}\n{'=' * 60}")
    loaders, scaler, metadata = build_dataloaders(
        crop,
        batch_size=TRAIN_HPARAMS["batch_size"],
        num_workers=TRAIN_HPARAMS["num_workers"],
    )

    if len(loaders["train"].dataset) == 0:
        raise RuntimeError(f"No training samples for {crop}")

    model, args = build_model(use_gpu=True)
    device = get_device(args)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_HPARAMS["learning_rate"])
    scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=TRAIN_HPARAMS["learning_rate"],
        steps_per_epoch=max(len(loaders["train"]), 1),
        epochs=TRAIN_HPARAMS["train_epochs"],
        pct_start=0.3,
    )
    early_stop = EarlyStopping(patience=TRAIN_HPARAMS["patience"])

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(TRAIN_HPARAMS["train_epochs"]):
        train_loss = _run_epoch(model, loaders["train"], criterion, optimizer, device, train=True)
        val_loss = _run_epoch(model, loaders["val"], criterion, optimizer, device, train=False)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"Epoch {epoch + 1:02d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        early_stop(val_loss, model)
        if early_stop.early_stop:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    if early_stop.best_state is not None:
        model.load_state_dict(early_stop.best_state)

    crop_model_dir = MODELS_DIR / crop.lower()
    crop_model_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = crop_model_dir / "checkpoint.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "crop": crop,
            "feature_cols": metadata["feature_columns"],
        },
        checkpoint_path,
    )

    scaler_path = crop_model_dir / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    config_path = crop_model_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "crop": crop,
                "model": "PatchTST",
                "seq_len": args.seq_len,
                "pred_len": args.pred_len,
                "label_len": args.label_len,
                "feature_columns": metadata["feature_columns"],
                "train_samples": len(loaders["train"].dataset),
                "val_samples": len(loaders["val"].dataset),
                "test_samples": len(loaders["test"].dataset),
                "best_val_loss": early_stop.best_loss,
                "final_train_loss": history["train_loss"][-1],
                "epochs_trained": len(history["train_loss"]),
            },
            f,
            indent=2,
        )

    out_dir = OUTPUTS_DIR / crop.lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    _plot_loss_curve(history, out_dir / "loss_curve.png")

    result = {
        "crop": crop,
        "checkpoint": str(checkpoint_path),
        "scaler": str(scaler_path),
        "best_val_loss": early_stop.best_loss,
        "final_train_loss": history["train_loss"][-1],
        "history": history,
    }
    print(f"Saved model for {crop} to {checkpoint_path}")
    return result


def _plot_loss_curve(history: dict, path: Path):
    plt.figure(figsize=(8, 4))
    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["val_loss"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("PatchTST Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def train_all_crops() -> dict:
    results = {}
    for crop in TARGET_CROPS:
        results[crop] = train_crop(crop)
    return results


if __name__ == "__main__":
    train_all_crops()
