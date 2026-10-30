"""
Full retrain + evaluate pipeline with improved hyperparameters.
Run from patchtst/ root:
    python run_training.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from agri_system.config import (
    SEQ_LEN, PRED_LEN, PATCHTST_HPARAMS, TRAIN_HPARAMS, TARGET_CROPS
)

print("=" * 60)
print("AgriField PatchTST — Improved Training Run")
print("=" * 60)
print(f"SEQ_LEN       : {SEQ_LEN}")
print(f"PRED_LEN      : {PRED_LEN}")
print(f"d_model       : {PATCHTST_HPARAMS['d_model']}")
print(f"n_heads       : {PATCHTST_HPARAMS['n_heads']}")
print(f"e_layers      : {PATCHTST_HPARAMS['e_layers']}")
print(f"d_ff          : {PATCHTST_HPARAMS['d_ff']}")
print(f"patch_len     : {PATCHTST_HPARAMS['patch_len']}")
print(f"stride        : {PATCHTST_HPARAMS['stride']}")
print(f"dropout       : {PATCHTST_HPARAMS['dropout']}")
print(f"train_epochs  : {TRAIN_HPARAMS['train_epochs']}")
print(f"patience      : {TRAIN_HPARAMS['patience']}")
print(f"learning_rate : {TRAIN_HPARAMS['learning_rate']}")
print(f"batch_size    : {TRAIN_HPARAMS['batch_size']}")
print(f"Crops         : {TARGET_CROPS}")
print("=" * 60)

# ── Step 1: Preprocess (needed because SEQ_LEN changed) ───────────────
print("\n[1/3] Preprocessing data...")
from agri_system.preprocess import run_preprocessing
run_preprocessing()
print("Preprocessing done.")

# ── Step 2: Train all crops ───────────────────────────────────────────
print("\n[2/3] Training models...")
from agri_system.train import train_all_crops
results = train_all_crops()
for crop, r in results.items():
    print(f"  {crop}: best_val_loss={r['best_val_loss']:.6f}")

# ── Step 3: Evaluate all crops ────────────────────────────────────────
print("\n[3/3] Evaluating models...")
from agri_system.evaluate import evaluate_all_crops
eval_results = evaluate_all_crops()

print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
for crop, r in eval_results.items():
    p = r["patchtst"]
    b = r["baseline"]
    print(f"{crop}:")
    print(f"  PatchTST  RMSE={p['RMSE']:.4f}  MAE={p['MAE']:.4f}  R²={p['R2']:.4f}")
    print(f"  Baseline  RMSE={b['RMSE']:.4f}  MAE={b['MAE']:.4f}  R²={b['R2']:.4f}")
    print(f"  RMSE improvement: {r['comparison']['rmse_improvement_pct']:.1f}%")

print("\nAll done. Restart the dashboard to see updated metrics.")
