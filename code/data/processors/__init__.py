"""Shared utilities for ACCV 2026 pipeline."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import yaml


def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str | Path) -> Dict:
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_split_df(
    splits_root: Path,
    split: str,
    merged_root: Path,
) -> pd.DataFrame:
    """
    Load video list for a given split (train / val / test).

    Parameters
    ----------
    splits_root : Path to datasets/splits/
    split       : "train" | "val" | "test"
    merged_root : Path to datasets/merged/

    Returns
    -------
    DataFrame with columns: video_path, label
    """
    txt_path = splits_root / f"{split}_videos.txt"
    if not txt_path.exists():
        raise FileNotFoundError(
            f"Split file not found: {txt_path}\n"
            f"Run: python datasets/validate_dataset.py --create-splits"
        )

    rows = []
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                parts = line.split(" ", 1)
            video_path = Path(parts[0])
            raw_label = parts[1].strip()

            # Normalize label: "violence"/"non_violence"/"1"/"0" → int 1/0
            if raw_label in ("violence", "1"):
                label = 1
            elif raw_label in ("non_violence", "0"):
                label = 0
            else:
                label = int(raw_label)

            if not video_path.is_absolute():
                video_path = merged_root / video_path

            rows.append({"video_path": str(video_path), "label": label})

    return pd.DataFrame(rows)
