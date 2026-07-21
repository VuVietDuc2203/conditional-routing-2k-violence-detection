"""
dataset_ver1_inference.py
========================
Dataset Processor — Ver 1: Inference / Evaluation
-------------------------------------------------
Produces a *pre-extracted frame dataset* suitable for running pretrained
model inference (C3D / I3D / SlowFast) WITHOUT any fine-tuning.

Design goals
~~~~~~~~~~~~
- Fast I/O  : all videos → pre-extracted frames on disk (or memmap)
- Model-agnostic : each sample = (N_frames, H, W, 3) numpy array
- Per-dataset or merged evaluation
- Reproducible : deterministic frame sampling via fixed seed

Output structure
~~~~~~~~~~~~~~~~
  ver1_inference/
  ├── clips/                    # one .npy per video clip
  │   ├── hockey_001.npy        # (16, 224, 224, 3) uint8
  │   ├── hockey_002.npy
  │   └── ...
  ├── metadata.csv              # index of all extracted clips
  └── stats.json                # class balance, fps, resolution stats

Usage
~~~~~
  python -m datasets.processors.dataset_ver1_inference \
      --config config.yaml \
      --split test \
      --model i3d \
      --merged \
      --resume

Author : ACCV 2026 Pipeline
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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
log = logging.getLogger("ver1_inference")


# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config(config_path: str) -> Dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_frame_size(model_name: str, cfg: Dict) -> Tuple[int, int]:
    return tuple(cfg["video"]["frame_size"].get(model_name, [224, 224]))


def resolve_clip_profile(model_name: str, cfg: Dict) -> str:
    """
    Resolve a shared extraction profile key.

    Models with identical extraction settings (sample_fps, clip_len, frame_size,
    extractor type) will reuse the same clip cache directory.
    """
    sample_fps = int(cfg["video"]["sample_fps"])
    if model_name == "slowfast":
        clip_len = int(cfg["video"]["slowfast_clip_len"])
        extractor = "slowfast"
    else:
        clip_len = int(cfg["video"]["clip_len"])
        extractor = "rgb"

    h, w = get_frame_size(model_name, cfg)
    return f"{extractor}_fps{sample_fps}_t{clip_len}_{h}x{w}"


def build_split_df(splits_root: Path, split: str, merged_root: Path) -> pd.DataFrame:
    """Load video list for the requested split."""
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
                parts = line.split(" ")
            video_path = Path(parts[0])
            raw_label = parts[1].strip()
            if raw_label in ("violence", "1"):
                label = 1
            elif raw_label in ("non_violence", "0"):
                label = 0
            else:
                label = int(raw_label)

            # Resolve absolute path relative to merged_root
            if not video_path.is_absolute():
                video_path = merged_root / video_path

            rows.append({"video_path": str(video_path), "label": label})

    df = pd.DataFrame(rows)
    log.info(f"Loaded {len(df)} videos for split='{split}': "
             f"violence={sum(df['label']==1)}, non_violence={sum(df['label']==0)}")
    return df


def extract_clips(
    video_path: str,
    sample_fps: int,
    clip_len: int,
    target_size: Tuple[int, int],
    max_frames: int,
    min_frames: int,
) -> List[np.ndarray]:
    """
    Sample `clip_len` frames uniformly from a video at `sample_fps`.

    Returns
    -------
    List of (clip_len, H, W, 3) uint8 arrays.
    One clip per `clip_len` window sliding by `clip_len//2` frames.
    If total_frames < min_frames → returns [].
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.warning(f"Cannot open video: {video_path}")
        return []

    fps       = cap.get(cv2.CAP_PROP_FPS)
    n_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if n_frames < min_frames:
        return []

    # Determine frame indices to extract
    effective_fps = min(fps, sample_fps) if fps > 0 else sample_fps
    frame_interval = max(1, round(fps / effective_fps)) if effective_fps else 1

    all_indices = list(range(0, n_frames, frame_interval))
    if len(all_indices) < min_frames:
        return []

    # Cap to max_frames for very long videos
    all_indices = all_indices[:max_frames]

    # Read selected frames
    frames = []
    cap = cv2.VideoCapture(video_path)
    for idx in all_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    if len(frames) < min_frames:
        return []

    # Build clips: sliding window
    clips = []
    step  = clip_len // 2   # 50% overlap
    for start in range(0, len(frames) - clip_len + 1, step):
        clip = np.stack(frames[start:start + clip_len], axis=0)   # (T, H, W, 3)
        clips.append(clip)

    return clips


def extract_slowfast_clips(
    video_path: str,
    sample_fps: int,
    clip_len: int,
    target_size: Tuple[int, int],
    max_frames: int,
    min_frames: int,
) -> List[Dict[str, np.ndarray]]:
    """
    SlowFast-style extraction: slow pathway (every 8th frame) + fast pathway.
    Returns list of dicts with keys 'slow' (8 frames) and 'fast' (32 frames).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.warning(f"Cannot open video: {video_path}")
        return []

    fps       = cap.get(cv2.CAP_PROP_FPS)
    n_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if n_frames < min_frames:
        return []

    effective_fps = min(fps, sample_fps) if fps > 0 else sample_fps
    frame_interval = max(1, round(fps / effective_fps)) if effective_fps else 1

    all_indices = list(range(0, n_frames, frame_interval))
    all_indices = all_indices[:max_frames]

    frames = []
    cap = cv2.VideoCapture(video_path)
    for idx in all_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    if len(frames) < min_frames:
        return []

    clips = []
    step  = clip_len // 2

    # Short videos: keep one padded clip instead of dropping everything
    if len(frames) < clip_len:
        window = list(frames)
        pad_count = clip_len - len(window)
        window.extend([window[-1]] * pad_count)
        fast = np.stack(window, axis=0)        # (clip_len, H, W, 3)
        slow = np.stack(window[::4], axis=0)   # (8, H, W, 3) for clip_len=32
        clips.append({"slow": slow, "fast": fast})
        return clips

    for start in range(0, len(frames) - clip_len + 1, step):
        window = frames[start:start + clip_len]

        # Slow: every 4th frame from window (8 frames)
        slow = np.stack(window[::4], axis=0)   # (8, H, W, 3)
        # Fast: all frames in window
        fast = np.stack(window, axis=0)        # (clip_len, H, W, 3)

        clips.append({"slow": slow, "fast": fast})

    return clips


def video_md5(video_path: str) -> str:
    """Quick hash for deduplication."""
    h = hashlib.md5()
    with open(video_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ──────────────────────────────────────────────────────────────────────────────
#  Main Extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_ver1(
    cfg: Dict,
    split: str,
    model_name: str,
    merged: bool = True,
    resume: bool = True,
    dry_run: bool = False,
) -> Path:
    """
    Run Ver 1 extraction for all videos in the given split.

    Parameters
    ----------
    cfg       : loaded config dict
    split     : "train" | "val" | "test"
    model_name: "c3d" | "i3d" | "slowfast" | "resnet" (controls frame size)
    merged    : if True, use merged/ dataset; else use per-dataset raw/ folders
    resume    : skip videos whose .npy already exists
    dry_run   : count videos but don't extract

    Returns
    -------
    Path to the output directory
    """
    merged_root = Path(cfg["paths"]["merged_root"])
    splits_root = Path(cfg["paths"]["splits_root"])
    output_root = Path(cfg["paths"]["output_root"]) / "ver1_inference"
    output_root.mkdir(parents=True, exist_ok=True)

    clip_profile = resolve_clip_profile(model_name, cfg)
    clip_dir  = output_root / "clips" / clip_profile / split
    clip_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Output directory: {clip_dir}")
    log.info(f"Clip profile: {clip_profile} (requested model={model_name})")
    set_seed(cfg["experiments"]["seed"])

    # ── Load split ──────────────────────────────────────────────
    df = build_split_df(splits_root, split, merged_root)

    # ── Extraction params ──────────────────────────────────────
    sample_fps  = cfg["video"]["sample_fps"]
    clip_len    = cfg["video"]["clip_len"]
    if model_name == "slowfast":
        clip_len = cfg["video"]["slowfast_clip_len"]
    target_size = get_frame_size(model_name, cfg)
    max_frames  = cfg["video"]["max_frames"]
    min_frames  = cfg["video"]["min_frames"]

    records = []
    skipped = 0
    extracted = 0

    # Quick filter for resume
    already_done = set()
    if resume:
        already_done = {p.stem for p in clip_dir.glob("*.npy")} | {p.stem for p in clip_dir.glob("*.npz")}
        log.info(f"Resume: {len(already_done)} clips already extracted")

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"[Ver1] {model_name}/{split}"):
        video_path = row["video_path"]
        label      = row["label"]
        video_hash = video_md5(video_path)

        if not Path(video_path).exists():
            log.warning(f"Video not found: {video_path}")
            continue

        if dry_run:
            records.append({"video_path": video_path, "label": label, "status": "dry_run"})
            continue

        # Build clip identifier (deterministic from path + clip index)
        base_name = Path(video_path).stem

        if model_name == "slowfast":
            clips = extract_slowfast_clips(
                video_path, sample_fps, clip_len, target_size, max_frames, min_frames
            )
            for ci, clip_data in enumerate(clips):
                clip_name = f"{base_name}_clip{ci:04d}"
                if resume and clip_name in already_done:
                    skipped += 1
                    continue
                # Save slow + fast as separate .npz
                out_path = clip_dir / f"{clip_name}.npz"
                np.savez_compressed(out_path, **clip_data)
                records.append({
                    "clip_path": str(out_path),
                    "video_path": video_path,
                    "clip_index": ci,
                    "label": label,
                    "video_hash": video_hash,
                    "model": model_name,
                    "clip_profile": clip_profile,
                    "clip_profile": clip_profile,
                    "split": split,
                })
                extracted += 1
        else:
            clips = extract_clips(
                video_path, sample_fps, clip_len, target_size, max_frames, min_frames
            )
            for ci, clip in enumerate(clips):
                clip_name = f"{base_name}_clip{ci:04d}"
                if resume and clip_name in already_done:
                    skipped += 1
                    continue
                out_path = clip_dir / f"{clip_name}.npy"
                np.save(out_path, clip)    # (.npy = (T,H,W,3))
                records.append({
                    "clip_path": str(out_path),
                    "video_path": video_path,
                    "clip_index": ci,
                    "label": label,
                    "video_hash": video_hash,
                    "model": model_name,
                    "clip_profile": clip_profile,
                    "clip_profile": clip_profile,
                    "split": split,
                })
                extracted += 1

    # ── Save metadata ───────────────────────────────────────────
    if not dry_run and records:
        meta_df = pd.DataFrame(records)
        meta_path = clip_dir / "metadata.csv"
        meta_df.to_csv(meta_path, index=False)
        log.info(f"Saved metadata: {meta_path}  ({len(meta_df)} clips)")

        # Class balance summary
        violence_count = sum(meta_df["label"] == 1)
        nonviolence_count = sum(meta_df["label"] == 0)
        stats = {
            "split": split,
            "model": model_name,
            "clip_profile": clip_profile,
            "total_clips": len(meta_df),
            "violence_clips": int(violence_count),
            "non_violence_clips": int(nonviolence_count),
            "ratio": round(violence_count / max(nonviolence_count, 1), 4),
            "skipped_existing": skipped,
            "extracted_this_run": extracted,
        }
        stats_path = output_root / f"stats_{clip_profile}_{split}.json"
        stats_path.write_text(json.dumps(stats, indent=2))
        log.info(f"Class balance: violence={violence_count}, non_violence={nonviolence_count}")

    log.info(f"[Ver1] Done — {extracted} new clips, {skipped} skipped (resume)")
    return clip_dir


# ──────────────────────────────────────────────────────────────────────────────
#  Dataset Class (for PyTorch DataLoader)
# ──────────────────────────────────────────────────────────────────────────────

class Ver1InferenceDataset:
    """
    In-memory dataset for Ver 1 clips.
    Loads all .npy/.npz from clip_dir on init.

    Use with:
        dataset = Ver1InferenceDataset(clip_dir="results/ver1_inference/clips/i3d/test")
        loader = DataLoader(dataset, batch_size=8, num_workers=4)
    """

    def __init__(
        self,
        clip_dir: str | Path,
        model_name: str = "i3d",
        transform=None,
        device: str = "cpu",
    ):
        self.clip_dir   = Path(clip_dir)
        self.model_name = model_name
        self.transform  = transform
        self.device     = device

        meta_path = self.clip_dir / "metadata.csv"
        if meta_path.exists():
            self.df = pd.read_csv(meta_path)
        else:
            # Fallback: scan clip_dir
            paths = sorted(self.clip_dir.glob("*.npy"))
            self.df = pd.DataFrame({"clip_path": [str(p) for p in paths]})

        log.info(f"Ver1InferenceDataset: {len(self.df)} clips from {self.clip_dir}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        row     = self.df.iloc[idx]
        path    = Path(row["clip_path"])
        label   = int(row["label"]) if "label" in row else -1

        if self.model_name == "slowfast":
            data  = np.load(path)
            slow  = data["slow"]   # (8, H, W, 3)
            fast  = data["fast"]   # (32, H, W, 3)
            # Convert to (C, T, H, W)
            slow_t = np.transpose(slow, (3, 0, 1, 2)).astype(np.float32) / 255.0
            fast_t = np.transpose(fast, (3, 0, 1, 2)).astype(np.float32) / 255.0
            clip   = {"slow": slow_t, "fast": fast_t}
        else:
            clip = np.load(path)          # (T, H, W, 3)
            clip = np.transpose(clip, (3, 0, 1, 2)).astype(np.float32) / 255.0  # (C,T,H,W)

        if self.transform:
            clip = self.transform(clip)

        return clip, label


# ──────────────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ver 1 — Pre-extract clips for inference")
    p.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    p.add_argument("--split",  default="test", choices=["train","val","test"],
                   help="Which split to extract")
    p.add_argument("--model",  default="i3d",
                   choices=["c3d","i3d","slowfast","resnet"],
                   help="Model type (determines frame size)")
    p.add_argument("--merged", action="store_true",
                   help="Use merged dataset (default). Use --no-merged for per-dataset)")
    p.add_argument("--no-merged", dest="merged", action="store_false")
    p.add_argument("--resume",  action="store_true", default=True,
                   help="Skip already-extracted clips [default: True]")
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--dry-run", action="store_true", help="Count videos without extracting")
    p.set_defaults(merged=True, resume=True)
    return p.parse_args()


if __name__ == "__main__":
    args  = parse_args()
    cfg   = load_config(args.config)
    out   = extract_ver1(
        cfg         = cfg,
        split       = args.split,
        model_name  = args.model,
        merged      = args.merged,
        resume      = args.resume,
        dry_run     = args.dry_run,
    )
    print(f"\n✅ Ver 1 extraction complete → {out}")
