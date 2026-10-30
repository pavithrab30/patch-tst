"""PatchTST model construction using the official repository implementation."""
from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import torch

from agri_system.config import FEATURE_COLS, LABEL_LEN, PATCHTST_HPARAMS, PRED_LEN, PROJECT_ROOT, SEQ_LEN

PATCHTST_ROOT = PROJECT_ROOT / "PatchTST_supervised"


def _ensure_patchtst_path() -> None:
    path_str = str(PATCHTST_ROOT)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def build_patchtst_args(use_gpu: bool = True) -> Namespace:
    enc_in = len(FEATURE_COLS)
    args = Namespace(
        model="PatchTST",
        seq_len=SEQ_LEN,
        label_len=LABEL_LEN,
        pred_len=PRED_LEN,
        enc_in=enc_in,
        dec_in=enc_in,
        c_out=enc_in,
        use_gpu=use_gpu and torch.cuda.is_available(),
        gpu=0,
        use_multi_gpu=False,
        devices="0",
        **PATCHTST_HPARAMS,
    )
    return args


def build_model(use_gpu: bool = True) -> tuple[torch.nn.Module, Namespace]:
    _ensure_patchtst_path()
    from models import PatchTST

    args = build_patchtst_args(use_gpu=use_gpu)
    model = PatchTST.Model(args).float()
    device = torch.device("cuda:0" if args.use_gpu else "cpu")
    model = model.to(device)
    return model, args


def get_device(args: Namespace) -> torch.device:
    return torch.device("cuda:0" if args.use_gpu else "cpu")
