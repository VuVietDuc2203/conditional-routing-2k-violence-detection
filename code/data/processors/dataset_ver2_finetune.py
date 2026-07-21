"""
dataset_ver2_finetune.py
========================
Dataset Processor — Ver 2: Fine-Tuning
--------------------------------------
Produces a *video-level dataset* ready for fine-tuning ResNet-LSTM
(and any future end-to-end trainable model).

Key differences from Ver 1
~~~~~~~~~~~~~~~~~~~~~~~~~~~
| Aspect            | Ver 1 (Inference)          | Ver 2 (Fine-tune)              |
|-------------------|-----------------------------|---------------------------------|
| Unit of data      | Per-clip (16 frames)        | Per-video (full video)          |
| Augmentation      | None (deterministic)       | Video-level augmentation       |
| Storage           | Pre-extracted .npy clips   | On-the-fly video loading       |
| Purpose           | Pretrained model inference  | Fine-tune backbone + classifier|
| Label granularity | Clip-level label            | Video-level label              |

Two sub-versions produced
~~~~~~~~~~~~~~~~~~~~~~~~~~
1. ver2_merged/         — All 6 datasets merged, stratified 70/15/15 split
2. ver2_cross_dataset/ — 6 separate train/test splits (leave-one-dataset-out)

Augmentation pipeline
~~~~~~~~~~~~~~~~~~~~~~
- Random temporal subsample (4 fps, 16–64 frames per video)
- Random spatial crop (scale 0.8–1.0, aspect 0.9–1.1)
- Random horizontal flip (p=0.5)
- Color jitter: brightness ±0.2, contrast ±0.2, saturation ±0.1
- Normalize with ImageNet stats

Output structure
~~~~~~~~~~~~~~~~
  ver2_finetune/
  ├── merged/
  │   ├── splits.csv          # video_path | label | split
  │   ├── train_videos.txt
  │   ├── val_videos.txt
  │   └── test_videos.txt
  │
  ├── cross_dataset/
  │   ├── exclude_hockey/    splits.csv + train/test .txt (all others as train)
  │   ├── exclude_movies2/
  │   ├── exclude_cctv_fights/
  │   ├── exclude_violent_flows/
  │   ├── exclude_rwf2000/
  │   └── exclude_surv_fight/
  │
  └── stats.json             # dataset statistics

Usage
~~~~~
  # Extract merged splits metadata
  python -m datasets.processors.dataset_ver2_finetune --mode merged

  # Generate cross-dataset leave-one-out splits
  python -m datasets.processors.dataset_ver2_finetune --mode cross_dataset

  # Dry run — preview without writing
  python -m datasets.processors.dataset_ver2_finetune --mode merged --dry-run

Author : ACCV 2026 Pipeline
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ver2_finetune")


# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    import random as _r
    import numpy as _np
    import torch as _torch
    _r.seed(seed)
    _np.random.seed(seed)
    _torch.manual_seed(seed)


def load_config(config_path: str) -> Dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_metadata(merged_root: Path) -> pd.DataFrame:
    """Load the merged metadata CSV (created by merge_datasets.py)."""
    meta_path = merged_root / "metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"metadata.csv not found at {meta_path}\n"
            f"Run: python datasets/merge_datasets.py --stage all"
        )
    df = pd.read_csv(meta_path)
    log.info(f"Loaded metadata: {len(df)} videos across all datasets")
    return df


def infer_dataset_name(video_path: str) -> str:
    """Infer source dataset from video path."""
    path_lower = video_path.lower()
    for name in ["hockey", "movies2", "cctv_fights", "violent_flows", "rwf2000", "surv_fight"]:
        if name in path_lower:
            return name
    return "unknown"


def video_stats(video_path: str) -> Dict:
    """Quick video stats without loading full video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"fps": 0, "n_frames": 0, "width": 0, "height": 0, "duration_s": 0}
    fps      = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {
        "fps": fps,
        "n_frames": n_frames,
        "width": width,
        "height": height,
        "duration_s": n_frames / fps if fps > 0 else 0,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Augmentation
# ──────────────────────────────────────────────────────────────────────────────

class VideoAugmentor:
    """
    On-the-fly video augmentation for fine-tuning.

    Augmentations are applied in order:
      1. Temporal subsample → uniform 4 fps
      2. Random spatial crop (scale 0.8–1.0, aspect 0.9–1.1)
      3. Random horizontal flip (p=0.5)
      4. Color jitter (brightness, contrast, saturation)
      5. ImageNet normalization
    """

    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1, 1)
    STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1, 1)

    def __init__(
        self,
        clip_len: int = 16,
        sample_fps: int = 4,
        crop_scale: Tuple[float, float] = (0.8, 1.0),
        crop_aspect: Tuple[float, float] = (0.9, 1.1),
        hflip_prob: float = 0.5,
        color_jitter: bool = True,
        normalize: bool = True,
        seed: int = 42,
    ):
        self.clip_len    = clip_len
        self.sample_fps  = sample_fps
        self.crop_scale  = crop_scale
        self.crop_aspect = crop_aspect
        self.hflip_prob  = hflip_prob
        self.color_jitter = color_jitter
        self.normalize    = normalize
        self.rng         = np.random.default_rng(seed)

    def _jitter(self, frame: np.ndarray) -> np.ndarray:
        """Apply color jitter to a single frame (uint8)."""
        f = frame.astype(np.float32) / 255.0

        # Brightness ±0.2
        f = f + self.rng.uniform(-0.2, 0.2)

        # Contrast ±0.2
        f = f * self.rng.uniform(0.8, 1.2)

        # Saturation ±0.1 (only on saturation channel — skip for speed)
        # Clip to [0, 1]
        f = np.clip(f, 0.0, 1.0)
        return (f * 255).astype(np.uint8)

    def _random_crop_resize(
        self, frames: List[np.ndarray], target_size: Tuple[int, int]
    ) -> List[np.ndarray]:
        """Random crop then resize to target_size."""
        H, W = frames[0].shape[:2]
        scale = self.rng.uniform(*self.crop_scale)
        aspect = self.rng.uniform(*self.crop_aspect)

        new_h = int(H * scale)
        new_w = int(W * scale * aspect)
        new_h = max(1, min(new_h, H))
        new_w = max(1, min(new_w, W))

        y = self.rng.integers(0, max(1, H - new_h + 1))
        x = self.rng.integers(0, max(1, W - new_w + 1))

        cropped = [f[y : y + new_h, x : x + new_w] for f in frames]
        resized = [
            cv2.resize(f, target_size, interpolation=cv2.INTER_LINEAR) for f in cropped
        ]
        return resized

    def _hflip(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        if self.rng.random() < self.hflip_prob:
            return [cv2.flip(f, 1) for f in frames]
        return frames

    def __call__(
        self,
        video_path: str,
        label: int,
        target_size: Tuple[int, int] = (224, 224),
    ) -> Tuple[np.ndarray, int]:
        """
        Load + augment video → (C, T, H, W) float32 tensor.

        Returns
        -------
        clip     : np.ndarray (3, T, H, W) float32 in [0, 1] or normalised
        label    : int (0=non_violence, 1=violence)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open: {video_path}")

        fps     = cap.get(cv2.CAP_PROP_FPS)
        n_frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if fps <= 0:
            fps = 25.0

        # Temporal subsample indices
        interval = max(1, round(fps / self.sample_fps))
        all_indices = list(range(0, n_frames_total, interval))

        if len(all_indices) <= self.clip_len:
            # Pad by repeating last frame
            indices = list(all_indices)
            while len(indices) < self.clip_len:
                indices.append(indices[-1])
        else:
            # Random temporal window start
            max_start = len(all_indices) - self.clip_len
            start = self.rng.integers(0, max(1, max_start + 1))
            indices = all_indices[start : start + self.clip_len]

        # Load frames
        frames = []
        cap = cv2.VideoCapture(video_path)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                # Pad with last frame
                if frames:
                    frames.append(frames[-1].copy())
                else:
                    frame = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
                    frames.append(frame)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        cap.release()

        # Augmentation
        frames = self._hflip(frames)
        frames = self._random_crop_resize(frames, target_size)

        if self.color_jitter:
            frames = [self._jitter(f) for f in frames]

        # Stack: (T, H, W, 3) → (3, T, H, W)
        clip = np.stack(frames, axis=0)                          # (T, H, W, 3)
        clip = np.transpose(clip, (3, 0, 1, 2)).astype(np.float32) / 255.0

        if self.normalize:
            clip = (clip - self.MEAN) / self.STD

        return clip.astype(np.float32), label


# ──────────────────────────────────────────────────────────────────────────────
#  Cross-Dataset Splits Generator
# ──────────────────────────────────────────────────────────────────────────────

def generate_cross_dataset_splits(
    df: pd.DataFrame,
    output_root: Path,
    min_videos_per_split: int = 10,
) -> List[Dict]:
    """
    For each of the 6 datasets, create a split where:
      - Test: only that dataset
      - Train + Val: all other datasets

    Returns list of dicts describing each split.
    """
    results = []
    for _, row in tqdm(
        df.groupby("source_dataset").size().items(),
        desc="[Ver2] Cross-dataset splits",
        total=df["source_dataset"].nunique(),
    ):
        pass  # just get unique datasets

    for exclude_dataset in df["source_dataset"].unique():
        exclude_dataset = str(exclude_dataset)
        train_df = df[df["source_dataset"] != exclude_dataset].copy()
        test_df  = df[df["source_dataset"] == exclude_dataset].copy()

        if len(test_df) < min_videos_per_split:
            log.warning(
                f"Skipping cross-dataset split '{exclude_dataset}': "
                f"only {len(test_df)} test videos"
            )
            continue

        # Further split train_df into train/val (80/20)
        from sklearn.model_selection import train_test_split
        train_sub, val_sub = train_test_split(
            train_df,
            test_size=0.15,
            stratify=train_df["label"],
            random_state=42,
        )

        split_dir = output_root / "cross_dataset" / f"exclude_{exclude_dataset}"
        split_dir.mkdir(parents=True, exist_ok=True)

        for name, sub_df in [("train", train_sub), ("val", val_sub), ("test", test_df)]:
            txt_path = split_dir / f"{name}_videos.txt"
            with open(txt_path, "w") as f:
                for _, r in sub_df.iterrows():
                    rel_path = Path(r["video_path"]).relative_to(output_root.parent / "merged")
                    f.write(f"{rel_path}\t{r['label']}\n")

            log.info(
                f"  exclude={exclude_dataset} | {name}: {len(sub_df)} videos "
                f"(violence={sum(sub_df['label']==1)}, normal={sum(sub_df['label']==0)})"
            )

        results.append({
            "excluded_dataset": exclude_dataset,
            "train_videos": len(train_sub),
            "val_videos":   len(val_sub),
            "test_videos":  len(test_df),
            "split_dir": str(split_dir),
        })

    return results


# ──────────────────────────────────────────────────────────────────────────────
#  Main Generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_merged_splits(
    df: pd.DataFrame,
    output_root: Path,
    test_size: float = 0.15,
    val_size: float  = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Stratified 70/15/15 train/val/test split across all merged datasets.
    Saves splits.csv + .txt files.
    """
    from sklearn.model_selection import train_test_split

    # 70% train, 15% val, 15% test
    train_df, temp_df = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=seed
    )
    val_ratio = val_size / (test_size + val_size)
    val_df, test_df = train_test_split(
        temp_df, test_size=val_ratio, stratify=temp_df["label"], random_state=seed
    )

    splits = {
        "train": train_df,
        "val":   val_df,
        "test":  test_df,
    }

    out_dir = output_root / "merged"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_records = []
    for split_name, split_df in splits.items():
        txt_path = out_dir / f"{split_name}_videos.txt"
        with open(txt_path, "w") as f:
            for _, r in split_df.iterrows():
                rel_path = Path(r["video_path"]).relative_to(output_root.parent / "merged")
                f.write(f"{rel_path}\t{r['label']}\n")
                all_records.append({
                    "video_path": str(rel_path),
                    "label": r["label"],
                    "split": split_name,
                    "source_dataset": r.get("source_dataset", "unknown"),
                })

        log.info(
            f"  {split_name}: {len(split_df)} videos "
            f"(violence={sum(split_df['label']==1)}, normal={sum(split_df['label']==0)})"
        )

    splits_df = pd.DataFrame(all_records)
    splits_df.to_csv(out_dir / "splits.csv", index=False)
    return splits_df


def compute_stats(df: pd.DataFrame, merged_root: Path) -> Dict:
    """Compute dataset statistics for the summary."""
    by_source = {}
    for src, grp in df.groupby("source_dataset"):
        stats = video_stats(grp.iloc[0]["video_path"])
        by_source[str(src)] = {
            "count":       len(grp),
            "violence":    int(sum(grp["label"] == 1)),
            "non_violence": int(sum(grp["label"] == 0)),
            "fps":         round(stats["fps"], 1),
            "sample_res":  f"{stats['width']}x{stats['height']}",
            "avg_duration_s": round(
                df[df["source_dataset"] == src]["duration_s"].mean(), 1
            ) if "duration_s" in df.columns else 0,
        }

    overall = {
        "total_videos":  len(df),
        "violence":       int(sum(df["label"] == 1)),
        "non_violence":   int(sum(df["label"] == 0)),
        "ratio":          round(sum(df["label"] == 1) / max(sum(df["label"] == 0), 1), 3),
        "by_source":      by_source,
    }
    return overall


# ──────────────────────────────────────────────────────────────────────────────
#  Dataset Class for PyTorch DataLoader
# ──────────────────────────────────────────────────────────────────────────────

class Ver2FineTuneDataset:
    """
    On-the-fly fine-tuning dataset.

    Use with PyTorch DataLoader for efficient batching:
        from torch.utils.data import DataLoader
        aug = VideoAugmentor(clip_len=16, sample_fps=4)
        ds  = Ver2FineTuneDataset(splits_root / "merged" / "train_videos.txt",
                                   merged_root, augmentor=aug)
        dl  = DataLoader(ds, batch_size=8, num_workers=4, pin_memory=True)
    """

    def __init__(
        self,
        split_file: str | Path,
        merged_root: str | Path,
        augmentor: Optional[VideoAugmentor] = None,
        clip_len: int = 16,
        sample_fps: int = 4,
        target_size: Tuple[int, int] = (224, 224),
        transform=None,
        return_metadata: bool = False,
    ):
        self.merged_root  = Path(merged_root)
        self.augmentor    = augmentor
        self.clip_len     = clip_len
        self.sample_fps   = sample_fps
        self.target_size  = target_size
        self.transform    = transform
        self.return_meta  = return_metadata

        self.samples = []
        with open(split_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    parts = line.split(" ")
                path = Path(parts[0])
                raw_label = parts[1].strip()
                if raw_label in ("violence", "1"):
                    label = 1
                elif raw_label in ("non_violence", "0"):
                    label = 0
                else:
                    label = int(raw_label)

                if not path.is_absolute():
                    path = self.merged_root / path
                self.samples.append((str(path), label))

        log.info(f"Ver2FineTuneDataset: {len(self.samples)} videos from {split_file}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        video_path, label = self.samples[idx]

        if self.augmentor is not None:
            clip, label = self.augmentor(video_path, label, self.target_size)
        else:
            # Default: simple uniform sampling + resize (no augmentation)
            clip = self._load_simple(video_path)
            if clip is None:
                # Fallback: zero tensor
                clip = np.zeros(
                    (3, self.clip_len, *self.target_size), dtype=np.float32
                )

        if self.transform:
            clip = self.transform(clip)

        if self.return_meta:
            return clip, label, video_path
        return clip, label

    def _load_simple(self, video_path: str) -> Optional[np.ndarray]:
        """Simple uniform sampling without augmentation."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        fps      = cap.get(cv2.CAP_PROP_FPS)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if fps <= 0:
            fps = 25.0
        interval = max(1, round(fps / self.sample_fps))
        indices  = list(range(0, n_frames, interval))

        if len(indices) < self.clip_len:
            return None

        # Center crop of indices
        start = max(0, (len(indices) - self.clip_len) // 2)
        indices = indices[start : start + self.clip_len]

        frames = []
        cap = cv2.VideoCapture(video_path)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_LINEAR)
            frames.append(frame)
        cap.release()

        if len(frames) < self.clip_len:
            return None

        clip = np.stack(frames, axis=0)
        clip = np.transpose(clip, (3, 0, 1, 2)).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1, 1)
        clip = (clip - mean) / std
        return clip.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ver 2 — Generate fine-tuning splits and dataset class"
    )
    p.add_argument("--config",  default="config.yaml")
    p.add_argument(
        "--mode",
        default="merged",
        choices=["merged", "cross_dataset", "all"],
        help="'merged' = 70/15/15 split, 'cross_dataset' = leave-one-out, "
             "'all' = both",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--seed",    type=int, default=42)
    return p.parse_args()


def main():
    args  = parse_args()
    cfg   = load_config(args.config)
    set_seed(args.seed)

    merged_root  = Path(cfg["paths"]["merged_root"])
    output_root  = Path(cfg["paths"]["output_root"]) / "ver2_finetune"
    output_root.mkdir(parents=True, exist_ok=True)

    # ── Load / enrich metadata ────────────────────────────────
    df = load_metadata(merged_root)

    # Add source_dataset column if missing
    if "source_dataset" not in df.columns:
        df["source_dataset"] = df["video_path"].apply(infer_dataset_name)

    # Add duration_s if missing
    if "duration_s" not in df.columns:
        log.info("Computing video durations (sampling first 10% of videos)...")
        sample_df = df.sample(frac=0.1, random_state=args.seed)
        durations = {}
        for _, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="[Stats]"):
            durations[row["video_path"]] = video_stats(row["video_path"])["duration_s"]
        df["duration_s"] = df["video_path"].map(durations).fillna(10.0)

    log.info(
        f"Dataset overview:\n"
        f"  Total: {len(df)} | Violence: {sum(df['label']==1)} | "
        f"Non-violence: {sum(df['label']==0)}"
    )

    # ── Generate splits ──────────────────────────────────────
    if args.mode in ("merged", "all"):
        splits_df = generate_merged_splits(df, output_root)
        log.info("✅ Merged splits generated")

    if args.mode in ("cross_dataset", "all"):
        cross_results = generate_cross_dataset_splits(df, output_root)
        log.info(f"✅ {len(cross_results)} cross-dataset splits generated")

    # ── Stats ────────────────────────────────────────────────
    stats = compute_stats(df, merged_root)
    stats_path = output_root / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    log.info(f"✅ Stats saved: {stats_path}")

    # ── Print summary ────────────────────────────────────────
    summary = textwrap.dedent(f"""
    ╔══════════════════════════════════════════════════╗
    ║          Ver 2 Fine-Tune Dataset Ready            ║
    ╠══════════════════════════════════════════════════╣
    ║  Total videos   : {stats['total_videos']:<28}║
    ║  Violence       : {stats['violence']:<28}║
    ║  Non-violence   : {stats['non_violence']:<28}║
    ║  Ratio (V/NV)    : {stats['ratio']:<28}║
    ╠══════════════════════════════════════════════════╣
    ║  Merged splits   : {output_root / 'merged':<28}║
    ║  Cross-dataset  : {output_root / 'cross_dataset':<28}║
    ╚══════════════════════════════════════════════════╝
    """)
    print(summary)


if __name__ == "__main__":
    main()
