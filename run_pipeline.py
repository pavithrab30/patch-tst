"""End-to-end pipeline: preprocess → train → evaluate."""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agri_system.evaluate import evaluate_all_crops
from agri_system.preprocess import run_preprocessing
from agri_system.train import train_all_crops


def main():
    start = time.time()
    print("Step 1/3: Preprocessing...")
    run_preprocessing()

    print("\nStep 2/3: Training PatchTST models...")
    train_all_crops()

    print("\nStep 3/3: Evaluation...")
    results = evaluate_all_crops()

    elapsed = time.time() - start
    print(f"\nPipeline complete in {elapsed / 60:.1f} minutes.")
    print("\nSummary:")
    for crop, res in results.items():
        p = res["patchtst"]
        b = res["baseline"]
        print(
            f"  {crop}: MAE={p['MAE']:.4f} RMSE={p['RMSE']:.4f} R²={p['R2']:.4f} "
            f"| Baseline RMSE={b['RMSE']:.4f}"
        )


if __name__ == "__main__":
    main()
