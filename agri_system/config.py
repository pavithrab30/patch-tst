"""Configuration for AgriField-Manipur PatchTST forecasting pipeline."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "dataset" / "AgriField-Manipur_v2.csv"
PROCESSED_DIR = PROJECT_ROOT / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
OUTPUTS_DIR = ARTIFACTS_DIR / "outputs"

TARGET_CROPS = ["Rice", "Maize", "Tomato"]

FEATURE_COLS = [
    "temperature",
    "humidity",
    "soil_moisture",
    "ph",
    "N",
    "P",
    "K",
    "rainfall",
]

SEQ_LEN = 30
LABEL_LEN = 7
PRED_LEN = 7

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

PATCHTST_HPARAMS = {
    "patch_len": 8,
    "stride": 4,
    "padding_patch": "end",
    "revin": 1,
    "affine": 0,
    "subtract_last": 0,
    "decomposition": 0,
    "kernel_size": 25,
    "individual": 0,
    "d_model": 128,
    "n_heads": 4,
    "e_layers": 2,
    "d_ff": 256,
    "dropout": 0.1,
    "fc_dropout": 0.1,
    "head_dropout": 0.0,
}

TRAIN_HPARAMS = {
    "batch_size": 32,
    "learning_rate": 1e-3,
    "train_epochs": 25,
    "patience": 5,
    "num_workers": 0,
    "random_seed": 2021,
}
