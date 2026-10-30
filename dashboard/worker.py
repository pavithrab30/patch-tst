"""
Inference worker — called as a subprocess by server.py.
Must be run from patchtst/ root:
    python dashboard/worker.py <crop> <field_id>
Prints a single JSON line to stdout.
"""
from __future__ import annotations
import sys, os, json, traceback

# Fix path — worker is at patchtst/dashboard/worker.py
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

def _safe_float(v):
    if v is None:
        return None
    try:
        f = float(v)
        return None if (f != f) else f
    except Exception:
        return None

def _clean(obj):
    import numpy as np
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if (v != v) else v
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, float) and (obj != obj):
        return None
    return obj

def main():
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Usage: worker.py <crop> <field_id>"}))
        sys.exit(1)

    crop     = sys.argv[1]
    field_id = sys.argv[2]

    try:
        # Import torch FIRST before anything else to avoid pandas crash
        import torch  # noqa: F401

        import pandas as pd
        from agri_system.config import FEATURE_COLS
        from agri_system.inference import forecast_field

        result = forecast_field(crop, field_id)
        ctx         = result["context"]
        forecast_df = result["forecast"]
        rec         = result["recommendation"]
        assessment  = result["assessment"]

        forecast_records = []
        for _, row in forecast_df.iterrows():
            rec_row = {"date": str(row["date"].date())}
            for col in FEATURE_COLS:
                rec_row[col] = _safe_float(row[col])
            forecast_records.append(rec_row)

        payload = {
            "ok": True,
            "context":        _clean(ctx),
            "forecast":       forecast_records,
            "assessment":     _clean(assessment),
            "recommendation": _clean(rec),
        }
    except Exception as e:
        payload = {"ok": False, "error": str(e), "trace": traceback.format_exc()}

    print(json.dumps(payload))

if __name__ == "__main__":
    main()
