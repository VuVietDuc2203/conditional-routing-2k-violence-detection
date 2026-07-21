"""
Build MoViNet preprocessed cache from JRTIP pipeline.

This script processes videos through the JRTIP YOLOv11n/ByteTrack/HDBSCAN
kinematic-gate preprocessing pipeline and saves cropped/processed clips as
torch tensors for MoViNet training.

The cache is stored under result/gpu_cache/ with three profiles:
- movinet_preprocessed_t16_224
- movinet_preprocessed_t32_224
- movinet_preprocessed_t64_224

Each profile contains:
- samples/*.pt: torch saved tensors with shape (C, T, H, W), dtype uint8
- manifest.csv: metadata for all samples

Root output contains:
- movinet_preprocess_stats.json: overall statistics

If YOLO/HDBSCAN dependencies are not available, the script falls back
to a deterministic whole-frame center crop/letterbox method and marks
preprocess_status='fallback_wholeframe' in the manifest.

CLI:
    python -m data.scripts.build_movinet_preprocess_cache --limit 2

Smoke test:
    python -m data.scripts.build_movinet_preprocess_cache --limit 2 --dry-run
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import cv2
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Check optional dependencies for full pipeline
try:
    import hdbscan  # type: ignore
except Exception:  # pragma: no cover
    hdbscan = None

try:
    from ultralytics import YOLO  # type: ignore
except Exception:  # pragma: no cover
    YOLO = None

# Project imports
try:
    from training_code.train_movinet_violence.preprocess.config import (
        PreprocessConfig,
        load_config,
    )
    from training_code.train_movinet_violence.preprocess.processor import (
        ExtractedStack,
        VideoStackProcessor,
    )
    from training_code.train_movinet_violence.preprocess.utils import (
        BBox,
        clamp_bbox_xywh,
        crop_and_resize,
        xyxy_to_xywh,
    )
    FULL_PIPELINE_AVAILABLE = True
except ImportError as e:
    logging.getLogger(__name__).warning(
        "MoViNet preprocess pipeline imports failed: %s. "
        "Full pipeline will be unavailable; fallback method will be used if dependencies are installed.",
        e,
    )
    FULL_PIPELINE_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build_movinet_preprocess_cache")

LABEL_TO_INT = {"non_violence": 0, "violence": 1, "0": 0, "1": 1}
LEGACY_CACHE_MARKERS = (
    "Ver1InferenceDataset",
    "extract_ver1",
    "results/ver1_inference",
    "results\\ver1_inference",
    "data/preprocessed",
    "data\\preprocessed",
    "datasets/preprocessed",
    "datasets\\preprocessed",
)


def json_default(obj: Any) -> Any:
    """Convert numpy/pandas scalar values to JSON-safe Python values."""
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def reject_legacy_path(path: str | Path, field_name: str) -> None:
    """Reject legacy cache roots for new JRTIP preprocessing."""
    normalized = str(path).replace("\\", "/")
    for marker in LEGACY_CACHE_MARKERS:
        if marker.replace("\\", "/") in normalized:
            raise ValueError(f"{field_name} points to legacy cache/data: {path}")


def require_result_path(path: str | Path, field_name: str) -> None:
    """Ensure generated outputs stay under result/."""
    reject_legacy_path(path, field_name)
    parts = [part.lower() for part in Path(path).parts]
    if "result" not in parts and (not parts or parts[0] != "result"):
        raise ValueError(f"{field_name} must be under result/: {path}")


def find_ffmpeg_bin(ffmpeg_bin: Optional[str] = None) -> str:
    """Resolve FFmpeg from explicit arg, FFMPEG_BIN, or PATH."""
    candidate = ffmpeg_bin or os.getenv("FFMPEG_BIN") or shutil.which("ffmpeg")
    if not candidate:
        raise FileNotFoundError(
            "ffmpeg not found. Add ffmpeg to PATH, set FFMPEG_BIN, or pass --ffmpeg-bin."
        )
    path = Path(candidate)
    if path.exists() or shutil.which(candidate):
        return str(candidate)
    raise FileNotFoundError(f"ffmpeg binary not found: {candidate}")


def normalize_label(raw_label: Any) -> Tuple[int, str]:
    """Return numeric label and canonical label name."""
    label_name = str(raw_label).strip()
    if label_name not in LABEL_TO_INT:
        raise ValueError(f"Unsupported label '{raw_label}'. Expected violence/non_violence or 0/1.")
    label_int = int(LABEL_TO_INT[label_name])
    if label_name == "0":
        label_name = "non_violence"
    elif label_name == "1":
        label_name = "violence"
    return label_int, label_name


def stable_video_id(row: pd.Series, row_index: int, video_path: Path) -> str:
    """Create a stable video id from split row metadata."""
    parts = [
        str(row.get("source_dataset", "")),
        str(row.get("label", "")),
        str(row.get("original_path", "")),
        str(row.get("merged_filename", "")),
        str(row.get("split", "")),
        str(video_path.name),
        str(row_index),
    ]
    digest = hashlib.sha1("|".join(parts).encode("utf-8", errors="ignore")).hexdigest()[:12]
    source = str(row.get("source_dataset", "unknown")).replace(" ", "_")
    stem = video_path.stem.replace(" ", "_")
    return f"{source}_{stem}_{digest}"


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash a generated cache file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def decode_frames_ffmpeg(
    video_path: Path,
    ffmpeg_bin: str,
    size: int,
    max_frames: int,
    sample_fps: float,
) -> np.ndarray:
    """Decode at most max_frames RGB frames as (T,H,W,3) uint8 using FFmpeg."""
    vf = (
        f"fps={sample_fps},"
        f"scale={size}:{size}:force_original_aspect_ratio=increase,"
        f"crop={size}:{size}"
    )
    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        vf,
        "-frames:v",
        str(max_frames),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed for {video_path}: {err}")

    frame_bytes = size * size * 3
    raw = np.frombuffer(proc.stdout, dtype=np.uint8)
    frame_count = len(raw) // frame_bytes
    if frame_count <= 0:
        raise RuntimeError(f"ffmpeg decoded zero frames for {video_path}")
    raw = raw[: frame_count * frame_bytes]
    return raw.reshape(frame_count, size, size, 3)


def letterbox_frames(
    frames: np.ndarray, target_size: int
) -> np.ndarray:
    """Letterbox frames to square (target_size x target_size)."""
    t, h, w, c = frames.shape
    if h == w == target_size:
        return frames
    # Create black canvas
    out = np.zeros((t, target_size, target_size, c), dtype=np.uint8)
    if h > w:
        new_h = target_size
        new_w = int(w * target_size / h)
        resized = np.array([
            cv2.resize(f, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            for f in frames
        ])
        x_offset = (target_size - new_w) // 2
        out[:, :, x_offset:x_offset + new_w, :] = resized
    else:
        new_w = target_size
        new_h = int(h * target_size / w)
        resized = np.array([
            cv2.resize(f, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            for f in frames
        ])
        y_offset = (target_size - new_h) // 2
        out[:, y_offset:y_offset + new_h, :, :] = resized
    return out


def center_crop_frames(
    frames: np.ndarray, target_size: int
) -> np.ndarray:
    """Center crop frames to square (target_size x target_size)."""
    t, h, w, c = frames.shape
    if h == w == target_size:
        return frames
    # Determine crop size (min dimension)
    crop_size = min(h, w)
    y0 = (h - crop_size) // 2
    x0 = (w - crop_size) // 2
    cropped = frames[:, y0:y0 + crop_size, x0:x0 + crop_size, :]
    # Resize to target
    resized = np.array([
        cv2.resize(f, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        for f in cropped
    ])
    return resized


def make_clip_tensor(frames: np.ndarray, clip_length: int) -> torch.Tensor:
    """Uniformly subsample or pad frames, returning uint8 tensor (C,T,H,W)."""
    total = int(frames.shape[0])
    if total >= clip_length:
        indices = np.linspace(0, total - 1, clip_length).round().astype(np.int64)
        clip = frames[indices]
    else:
        pad = np.repeat(frames[-1:,:,:,:], clip_length - total, axis=0)
        clip = np.concatenate([frames, pad], axis=0)
    clip = np.ascontiguousarray(np.transpose(clip, (3, 0, 1, 2)))
    return torch.from_numpy(clip)


PERSON_MODEL = None


def get_person_model(model_name: str):
    """Load YOLOv11n person detector once per process."""
    global PERSON_MODEL
    if PERSON_MODEL is None:
        if YOLO is None:
            raise RuntimeError("ultralytics is required for YOLOv11n person detection.")
        PERSON_MODEL = YOLO(model_name, task="detect")
    return PERSON_MODEL


def bbox_iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """IoU for two xyxy boxes."""
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def expand_xyxy(box: np.ndarray, scale: float, width: int, height: int) -> np.ndarray:
    """Expand an xyxy box around its center and clamp to image size."""
    x1, y1, x2, y2 = [float(x) for x in box]
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    bw = max(1.0, x2 - x1) * float(scale)
    bh = max(1.0, y2 - y1) * float(scale)
    side = max(bw, bh)
    nx1 = max(0, int(round(cx - side * 0.5)))
    ny1 = max(0, int(round(cy - side * 0.5)))
    nx2 = min(width, int(round(cx + side * 0.5)))
    ny2 = min(height, int(round(cy + side * 0.5)))
    return np.array([nx1, ny1, nx2, ny2], dtype=np.float32)


def crop_xyxy_rgb(frame_rgb: np.ndarray, box: np.ndarray, size: int) -> np.ndarray:
    """Crop an RGB frame by xyxy and resize to square."""
    h, w = frame_rgb.shape[:2]
    box = expand_xyxy(box, 1.0, w, h).astype(int)
    crop = frame_rgb[box[1]:box[3], box[0]:box[2]]
    if crop.size == 0:
        crop = frame_rgb
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)


def rect_area_xyxy(box: np.ndarray) -> float:
    return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))


def overlap_min_area_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """Match ANSCustomViolence::CalculateOverlap: intersection / min(area_a, area_b)."""
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    min_area = min(rect_area_xyxy(a), rect_area_xyxy(b))
    return float(inter / min_area) if min_area > 0 else 0.0


def center_xyxy(box: np.ndarray) -> np.ndarray:
    return np.array([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5], dtype=np.float32)


def clamp_xyxy(box: np.ndarray, width: int, height: int) -> np.ndarray:
    return np.array(
        [
            max(0, min(width, int(round(float(box[0]))))),
            max(0, min(height, int(round(float(box[1]))))),
            max(0, min(width, int(round(float(box[2]))))),
            max(0, min(height, int(round(float(box[3]))))),
        ],
        dtype=np.float32,
    )


def cluster_person_indices(
    boxes_xyxy: np.ndarray,
    cluster_min_pts: int,
    cluster_selection_epsilon: float = 0.0,
) -> List[List[int]]:
    """Return crowd clusters as person-box indices."""
    if len(boxes_xyxy) < int(cluster_min_pts):
        return []

    centers = np.column_stack(
        [
            (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) * 0.5,
            (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) * 0.5,
        ]
    )

    if hdbscan is None or len(boxes_xyxy) < max(2, int(cluster_min_pts)):
        return [list(range(len(boxes_xyxy)))]

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=max(2, int(cluster_min_pts)),
        min_samples=1,
        cluster_selection_epsilon=max(0.0, float(cluster_selection_epsilon)),
    )
    labels = clusterer.fit_predict(centers)
    clusters: List[List[int]] = []
    for label in sorted(set(int(x) for x in labels if int(x) >= 0)):
        idxs = [idx for idx, lab in enumerate(labels) if int(lab) == label]
        if len(idxs) >= int(cluster_min_pts):
            clusters.append(idxs)
    return clusters


def union_xyxy(boxes_xyxy: np.ndarray) -> np.ndarray:
    return np.array(
        [
            float(boxes_xyxy[:, 0].min()),
            float(boxes_xyxy[:, 1].min()),
            float(boxes_xyxy[:, 2].max()),
            float(boxes_xyxy[:, 3].max()),
        ],
        dtype=np.float32,
    )


def cluster_person_boxes(
    boxes_xyxy: np.ndarray,
    cluster_min_pts: int,
    cluster_selection_epsilon: float = 0.0,
) -> List[np.ndarray]:
    """Return crowd candidate boxes using HDBSCAN over person centers."""
    if len(boxes_xyxy) == 0:
        return []
    if len(boxes_xyxy) < int(cluster_min_pts):
        return []

    centers = np.column_stack(
        [
            (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) * 0.5,
            (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) * 0.5,
        ]
    )

    if hdbscan is None or len(boxes_xyxy) < max(2, int(cluster_min_pts)):
        return [np.array([boxes_xyxy[:, 0].min(), boxes_xyxy[:, 1].min(), boxes_xyxy[:, 2].max(), boxes_xyxy[:, 3].max()])]

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=max(2, int(cluster_min_pts)),
        min_samples=1,
        cluster_selection_epsilon=max(0.0, float(cluster_selection_epsilon)),
    )
    labels = clusterer.fit_predict(centers)
    crowd_boxes: List[np.ndarray] = []
    for label in sorted(set(int(x) for x in labels if int(x) >= 0)):
        member = boxes_xyxy[labels == label]
        if len(member) >= int(cluster_min_pts):
            crowd_boxes.append(
                np.array([member[:, 0].min(), member[:, 1].min(), member[:, 2].max(), member[:, 3].max()])
            )
    return crowd_boxes


def match_crowd_track(
    curr_box: np.ndarray,
    curr_person_ids: List[int],
    available_tracks: List[Dict[str, Any]],
    track_used: List[bool],
    use_fallback_thresholds: bool,
) -> Tuple[int, float]:
    """Mirror the C++ hybrid crowd ID matching score."""
    match_iou_threshold = 0.05 if use_fallback_thresholds else 0.15
    match_dist_threshold = 400.0 if use_fallback_thresholds else 250.0
    curr_set = set(int(pid) for pid in curr_person_ids if int(pid) >= 0)

    best_idx = -1
    best_score = -1.0
    for idx, track in enumerate(available_tracks):
        if track_used[idx]:
            continue

        prev_ids = set(int(pid) for pid in track.get("person_ids", []) if int(pid) >= 0)
        person_overlap = len(curr_set & prev_ids)
        score = -1.0

        if person_overlap >= 2:
            score = 1.0
        elif person_overlap >= 1 and overlap_min_area_xyxy(curr_box, track["bbox"]) >= 0.10:
            score = 0.9
        else:
            overlap = overlap_min_area_xyxy(curr_box, track["bbox"])
            if overlap >= match_iou_threshold:
                dist = float(np.linalg.norm(center_xyxy(curr_box) - center_xyxy(track["bbox"])))
                if dist < match_dist_threshold:
                    score = 0.5 if use_fallback_thresholds else 0.8

        if score > best_score:
            best_score = score
            best_idx = idx

    return best_idx, best_score


def build_movinet_sequence_cpp_like(
    buffer: List[Dict[str, Any]],
    frame_width: int,
    frame_height: int,
    size: int,
    sequence_frames: int = 16,
) -> np.ndarray:
    """Build a crop sequence using ANSCustomViolence.cpp crop semantics."""
    sequence_frames = max(1, int(sequence_frames))
    crowd_boxes = [snap["crowd_box"] for snap in buffer if rect_area_xyxy(snap["crowd_box"]) > 0]
    if not crowd_boxes:
        frames = [cv2.resize(snap["frame"], (size, size), interpolation=cv2.INTER_AREA) for snap in buffer[:sequence_frames]]
        return np.stack(frames, axis=0)

    widths = [float(box[2] - box[0]) for box in crowd_boxes]
    heights = [float(box[3] - box[1]) for box in crowd_boxes]
    max_w = max(1.0, max(widths))
    max_h = max(1.0, max(heights))
    expanded_w = max_w * 2.0
    expanded_h = max_h * 2.0

    centers: List[Optional[np.ndarray]] = []
    last_center: Optional[np.ndarray] = None
    for snap in buffer:
        box = snap["crowd_box"]
        if rect_area_xyxy(box) > 0:
            last_center = center_xyxy(box)
        centers.append(None if last_center is None else last_center.copy())

    if centers and centers[0] is None:
        first_center = next((ctr for ctr in centers if ctr is not None), None)
        if first_center is not None:
            for idx, ctr in enumerate(centers):
                if ctr is None:
                    centers[idx] = first_center.copy()
                else:
                    break

    fallback_center = center_xyxy(crowd_boxes[-1])
    cropped: List[np.ndarray] = []
    denom = max(1, sequence_frames - 1)
    for seq_idx in range(sequence_frames):
        buffer_pos = (float(seq_idx) / float(denom)) * (len(buffer) - 1)
        frame_idx = int(buffer_pos)
        snap = buffer[frame_idx]
        ctr = centers[frame_idx] if centers[frame_idx] is not None else fallback_center

        x1 = int(round(float(ctr[0]) - expanded_w / 2.0))
        y1 = int(round(float(ctr[1]) - expanded_h / 2.0))
        x2 = int(round(float(ctr[0]) + expanded_w / 2.0))
        y2 = int(round(float(ctr[1]) + expanded_h / 2.0))
        crop_box = clamp_xyxy(np.array([x1, y1, x2, y2], dtype=np.float32), frame_width, frame_height).astype(int)
        crop = snap["frame"][crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]
        if crop.size == 0:
            crop = snap["frame"]
        interp = cv2.INTER_AREA if crop.shape[1] > size or crop.shape[0] > size else cv2.INTER_LANCZOS4
        cropped.append(cv2.resize(crop, (size, size), interpolation=interp))

    return np.stack(cropped, axis=0)


def process_video_jrtip_kinematic(video_path: Path, args: argparse.Namespace) -> Tuple[np.ndarray, Dict[str, Any], str]:
    """
    New JRTIP Stage-1 preprocess:
    YOLOv11n person detection + ByteTrack + HDBSCAN + kinematic gate.

    Kinematic gate replaces fightOD only as the trigger. After the trigger, this
    follows ANSCustomViolence.cpp stack/crop behavior: collect contiguous frames,
    keep tracking state for disappeared crowds, and build 16-frame MoViNet crops.
    """
    model = get_person_model(str(args.person_model))
    target_sequence_frames = max(int(x) for x in args.clip_lengths)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    all_sampled_frames: List[np.ndarray] = []
    full_clip_snapshots: List[Dict[str, Any]] = []
    prev_crowd_tracks: List[Dict[str, Any]] = []
    lost_crowd_tracks: List[Dict[str, Any]] = []
    prev_person_tracks: List[Dict[str, Any]] = []
    consecutive_person_kinetic_count: Dict[int, int] = {}
    next_crowd_track_id = 1
    next_person_track_id = 1
    sampled = 0
    person_frames = 0
    person_kinetic_frames = 0
    person_kinetic_pass = False
    crowd_frames = 0
    gate_frames = 0
    last_best_crowd_box: Optional[np.ndarray] = None
    fallback_reason: Optional[str] = None

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            fps = 25.0
        inv_sample = 1.0 / float(args.sample_fps)
        next_sample_t = 0.0
        frame_idx = 0

        while True:
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                break
            t = frame_idx / fps
            if t + 1e-9 < next_sample_t:
                frame_idx += 1
                continue
            next_sample_t += inv_sample
            sampled += 1

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            all_sampled_frames.append(cv2.resize(frame_rgb, (args.size, args.size), interpolation=cv2.INTER_LINEAR))
            frame_h, frame_w = frame_rgb.shape[:2]

            result = model.track(
                source=frame_bgr,
                persist=True,
                tracker=str(args.tracker),
                conf=float(args.person_conf),
                classes=[0],
                device=str(args.detector_device),
                verbose=False,
                imgsz=int(args.detector_imgsz),
                half=bool(args.half),
                stream=False,
            )[0]

            crowd_boxes: List[np.ndarray] = []
            crowd_person_ids_list: List[List[int]] = []
            if result.boxes is None or result.boxes.xyxy is None or len(result.boxes) == 0:
                boxes = np.empty((0, 4), dtype=np.float32)
                person_ids: List[int] = []
            else:
                boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
                if result.boxes.id is not None:
                    person_ids = result.boxes.id.detach().cpu().numpy().astype(int).tolist()
                else:
                    person_ids = [-1] * len(boxes)

            previous_person_tracks = prev_person_tracks
            updated_person_tracks: List[Dict[str, Any]] = []
            used_prev_person = [False] * len(previous_person_tracks)
            for box_idx, box in enumerate(boxes):
                curr_person_id = int(person_ids[box_idx]) if box_idx < len(person_ids) else -1
                best_idx = -1
                best_score = -1.0
                for prev_idx, prev_track in enumerate(previous_person_tracks):
                    if used_prev_person[prev_idx]:
                        continue
                    prev_person_id = int(prev_track.get("person_id", -1))
                    if curr_person_id >= 0 and prev_person_id == curr_person_id:
                        score = 2.0
                    else:
                        iou = bbox_iou_xyxy(prev_track["bbox"], box)
                        dist = float(np.linalg.norm(center_xyxy(prev_track["bbox"]) - center_xyxy(box)))
                        diag = max(1.0, float(np.linalg.norm([box[2] - box[0], box[3] - box[1]])))
                        score = iou if dist / diag <= float(args.person_match_dist) else -1.0
                    if score > best_score:
                        best_score = score
                        best_idx = prev_idx

                if best_idx >= 0 and best_score > 0:
                    prev_track = previous_person_tracks[best_idx]
                    used_prev_person[best_idx] = True
                    track_id = int(prev_track["track_id"])
                    iou = bbox_iou_xyxy(prev_track["bbox"], box)
                    diag = max(1.0, float(np.linalg.norm([box[2] - box[0], box[3] - box[1]])))
                    velocity_norm = float(np.linalg.norm(center_xyxy(box) - center_xyxy(prev_track["bbox"])) / diag)
                    kinetic = iou <= float(args.iou_gate) or velocity_norm >= float(args.velocity_gate)
                    if kinetic:
                        consecutive_person_kinetic_count[track_id] = consecutive_person_kinetic_count.get(track_id, 0) + 1
                    else:
                        consecutive_person_kinetic_count[track_id] = 0
                    if consecutive_person_kinetic_count[track_id] >= int(args.kappa_frames):
                        person_kinetic_pass = True
                        person_kinetic_frames += 1
                else:
                    track_id = next_person_track_id
                    next_person_track_id += 1
                    consecutive_person_kinetic_count[track_id] = 0

                updated_person_tracks.append(
                    {
                        "track_id": int(track_id),
                        "person_id": int(curr_person_id),
                        "bbox": box.copy(),
                    }
                )
            prev_person_tracks = updated_person_tracks

            if len(boxes) > 0:
                person_frames += 1
                for idxs in cluster_person_indices(boxes, int(args.cluster_min_pts), float(args.hdbscan_epsilon)):
                    cluster_boxes = boxes[idxs]
                    crowd_boxes.append(union_xyxy(cluster_boxes))
                    crowd_person_ids_list.append([int(person_ids[i]) for i in idxs if int(person_ids[i]) >= 0])

            group_track_ids: List[int] = []
            best_frame_crowd_box: Optional[np.ndarray] = None

            if crowd_boxes:
                crowd_frames += 1
                best_frame_crowd_box = max(
                    crowd_boxes,
                    key=lambda box: max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1])),
                )
                last_best_crowd_box = best_frame_crowd_box.copy()
                available_tracks: List[Dict[str, Any]] = []
                for prev in prev_crowd_tracks:
                    available_tracks.append({**prev, "is_lost": False})
                for lost in lost_crowd_tracks:
                    if int(lost["frames_since_lost"]) <= int(args.crowd_retain_frames):
                        available_tracks.append(
                            {
                                "track_id": lost["track_id"],
                                "bbox": lost["last_bbox"],
                                "person_ids": lost["last_person_ids"],
                                "is_lost": True,
                            }
                        )

                track_used = [False] * len(available_tracks)
                assignment: Dict[int, int] = {}
                crowd_indices = sorted(
                    range(len(crowd_boxes)),
                    key=lambda idx: len(crowd_person_ids_list[idx]),
                    reverse=True,
                )

                for g_idx in crowd_indices:
                    best_idx, best_score = match_crowd_track(
                        crowd_boxes[g_idx],
                        crowd_person_ids_list[g_idx],
                        available_tracks,
                        track_used,
                        use_fallback_thresholds=False,
                    )
                    if best_idx < 0 or best_score < 0:
                        best_idx, best_score = match_crowd_track(
                            crowd_boxes[g_idx],
                            crowd_person_ids_list[g_idx],
                            available_tracks,
                            track_used,
                            use_fallback_thresholds=True,
                        )

                    if best_idx >= 0 and best_score > 0:
                        assignment[g_idx] = int(available_tracks[best_idx]["track_id"])
                        track_used[best_idx] = True
                    else:
                        assignment[g_idx] = next_crowd_track_id
                        next_crowd_track_id += 1

                group_track_ids = [assignment[idx] for idx in range(len(crowd_boxes))]

                matched_ids = set(group_track_ids)
                lost_crowd_tracks = [lost for lost in lost_crowd_tracks if int(lost["track_id"]) not in matched_ids]
                prev_crowd_tracks = [
                    {
                        "track_id": int(tid),
                        "bbox": crowd_boxes[idx],
                        "person_ids": crowd_person_ids_list[idx],
                    }
                    for idx, tid in enumerate(group_track_ids)
                ]
            else:
                for prev in prev_crowd_tracks:
                    if not any(int(lost["track_id"]) == int(prev["track_id"]) for lost in lost_crowd_tracks):
                        lost_crowd_tracks.append(
                            {
                                "track_id": int(prev["track_id"]),
                                "last_bbox": prev["bbox"],
                                "last_person_ids": prev["person_ids"],
                                "frames_since_lost": 0,
                            }
                        )

            for lost in lost_crowd_tracks:
                lost["frames_since_lost"] = int(lost["frames_since_lost"]) + 1
            lost_crowd_tracks = [
                lost for lost in lost_crowd_tracks
                if int(lost["frames_since_lost"]) <= int(args.crowd_retain_frames)
            ]

            if person_kinetic_pass:
                gate_frames += 1

            full_clip_snapshots.append(
                {
                    "frame": frame_rgb.copy(),
                    "crowd_box": (
                        best_frame_crowd_box.copy()
                        if best_frame_crowd_box is not None
                        else (last_best_crowd_box.copy() if last_best_crowd_box is not None else np.zeros(4, dtype=np.float32))
                    ),
                    "has_crowd": best_frame_crowd_box is not None,
                }
            )

            frame_idx += 1
    finally:
        cap.release()

    gate_params = {
        "method": "yolo11n_bytetrack_hdbscan_kinematic_cpp_stack",
        "person_model": str(args.person_model),
        "detector_device": str(args.detector_device),
        "person_conf": float(args.person_conf),
        "tracker": str(args.tracker),
        "cluster_min_pts": int(args.cluster_min_pts),
        "iou_gate": float(args.iou_gate),
        "velocity_gate": float(args.velocity_gate),
        "kappa_frames": int(args.kappa_frames),
        "crowd_retain_frames": int(args.crowd_retain_frames),
        "movinet_sequence_frames": int(target_sequence_frames),
        "movinet_crop_scale": 2.0,
        "legacy_crop_scale_arg": float(args.crop_scale),
        "sample_fps": float(args.sample_fps),
        "sampled_frames": int(sampled),
        "person_frames": int(person_frames),
        "person_kinetic_frames": int(person_kinetic_frames),
        "crowd_frames": int(crowd_frames),
        "gate_frames": int(gate_frames),
        "completed_stacks": 0,
    }

    if person_kinetic_pass and len(full_clip_snapshots) >= min(int(target_sequence_frames), len(full_clip_snapshots)):
        sequence = build_movinet_sequence_cpp_like(
            full_clip_snapshots[:target_sequence_frames],
            frame_w,
            frame_h,
            int(args.size),
            target_sequence_frames,
        )
        gate_params["completed_stacks"] = 1
        return sequence, gate_params, "success"
    if all_sampled_frames:
        fallback_reason = fallback_reason or "no_person_kinetic_pass"
        gate_params["fallback_reason"] = fallback_reason
        return np.stack(all_sampled_frames, axis=0), gate_params, "fallback_wholeframe"
    raise RuntimeError(f"No sampled frames decoded for {video_path}")


def aggregate_label_from_stacks(stacks: List[ExtractedStack]) -> str:
    """Determine aggregate label from multiple stacks by majority vote of frame-level flags."""
    if not stacks:
        return "normal"
    # Count violence frames across all stacks
    violence_frames = 0
    total_frames = 0
    for stack in stacks:
        # Each stack has label determined by fight detection
        # We can't easily recover per-frame flags from ExtractedStack
        # So we use the stack label as a proxy: if stack.label == "violence", count all frames as violence
        # This is approximate but consistent
        label = stack.label
        # We don't know frame count in stack.frames list length is the count
        count = len(stack.frames)
        total_frames += count
        if label == "violence":
            violence_frames += count
    # Apply same threshold as processor: need at least 2 violence-equivalent frames
    # Since we're counting whole stacks, if any stack is violence, we count its frames as violence
    # The threshold in processor is: fight_frame_count >= valid_violence_frames (default 2)
    # For a 16-frame stack, it's all-or-nothing: either >=2 frames triggered violence -> stack is violence
    # So for combining stacks, we can count how many stacks are violence and require at least 1 for 16,
    # at least 2 for 32? Actually the rule should be consistent: out of T frames, need >=2 violent frames.
    # For 16 frames from 1 stack: if that stack is violence, it already has >=2 violent frames.
    # For 32 frames from 2 stacks: if both stacks are violence -> definitely >=2; if one stack violence -> that stack has >=2, so overall >=2.
    # So if ANY stack is violence, the combined clip has at least 2 violent frames.
    # Therefore: if any stack.label == "violence", return "violence".
    for stack in stacks:
        if stack.label == "violence":
            return "violence"
    return "normal"


class MovinetPreprocessProcessor:
    """Wrapper around VideoStackProcessor that captures stacks instead of writing videos.

    This adapter allows us to integrate the existing YOLO/ByteTrack/HDBSCAN pipeline
    while generating cache for MoViNet without writing intermediate video files.
    """

    def __init__(
        self,
        cfg: PreprocessConfig,
        crop_resize_size: int = 224,
    ):
        self.cfg = cfg
        self.crop_resize_size = crop_resize_size
        self.captured_stacks: List[ExtractedStack] = []
        self._video_path: Optional[Path] = None
        self._video_stem: str = ""

        # Override frozen config settings for cache building.
        self.cfg = replace(
            self.cfg,
            runtime=replace(self.cfg.runtime, show=False, save_debug=False),
            stacking=replace(self.cfg.stacking, crop_resize_size=int(crop_resize_size)),
        )

        # Check dependencies
        if YOLO is None:
            raise RuntimeError(
                "ultralytics (YOLO) is required for full pipeline. "
                "Install with: pip install ultralytics"
            )
        if hdbscan is None:
            raise RuntimeError(
                "hdbscan is required for full pipeline. "
                "Install with: pip install hdbscan"
            )

        # Initialize processor with models
        self.processor = VideoStackProcessor(cfg)

        # Override _finalize_stack to capture instead of save video
        self._original_finalize = self.processor._finalize_stack
        self.processor._finalize_stack = self._capture_finalize_stack

    def _capture_finalize_stack(self, stack) -> ExtractedStack:
        """Override that captures the finalized stack."""
        # Call original to get the ExtractedStack (which includes fight detection label)
        result = self._original_finalize(stack)

        # Convert BGR to RGB (frames already at target size from config)
        if result.frames:
            rgb_frames = []
            for f in result.frames:
                rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                rgb_frames.append(rgb)
            result.frames = rgb_frames

        self.captured_stacks.append(result)
        return result

    def process_video(self, video_path: Path) -> List[ExtractedStack]:
        """Process a video and return all captured stacks."""
        self.captured_stacks.clear()
        self._video_path = video_path
        self._video_stem = video_path.stem

        # Use the processor's main processing loop
        self.processor.reset_state()
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if fps <= 0:
                fps = 25.0

            sample_fps = float(self.cfg.video.sample_fps)
            inv_sample = 1.0 / sample_fps
            next_sample_t = 0.0

            frame_idx = 0
            track_stride = max(1, int(getattr(self.cfg.video, "track_stride", 1)))
            track_every_frame = bool(getattr(self.cfg.video, "track_every_frame", True))

            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                # Update person tracking on raw frames
                if track_every_frame and self.cfg.tracking.use_bytetrack and (frame_idx % track_stride == 0):
                    self.processor.update_person_tracks(frame)

                t = frame_idx / fps
                if t + 1e-9 >= next_sample_t:
                    # Process sampled frame
                    res = self.processor.process_sampled_frame(frame, camera_id=self._video_stem)
                    # Stacks are captured via overridden _finalize_stack
                    next_sample_t += inv_sample

                frame_idx += 1
        finally:
            cap.release()
            self.processor.reset_state()

        return self.captured_stacks.copy()


def check_dependencies() -> Tuple[bool, List[str]]:
    """Check if full pipeline dependencies are available."""
    missing = []
    if YOLO is None:
        missing.append("ultralytics (YOLO)")
    if hdbscan is None:
        missing.append("hdbscan")
    return (len(missing) == 0, missing)


def missing_model_paths(config: PreprocessConfig) -> List[str]:
    """Return configured local model paths that do not exist."""
    missing: List[str] = []
    for path_text in [config.models.person_detector_onnx, config.models.fight_detector_onnx]:
        path = Path(path_text)
        if path.suffix and not path.exists():
            missing.append(str(path))
    return missing


def profile_name(preprocess_type: str, clip_length: int, size: int) -> str:
    return f"{preprocess_type}_t{clip_length}_{size}"


def process_one(
    row_index: int,
    row_dict: Dict[str, Any],
    args: argparse.Namespace,
    config: Optional[PreprocessConfig] = None,
) -> Dict[str, Any]:
    """Process one split row and return manifest records plus status."""
    row = pd.Series(row_dict)

    # Resolve video path
    merged_root = Path(args.merged_root)
    label = str(row.get("label", "")).strip()
    merged_filename = str(row.get("merged_filename", "")).strip()
    merged_path = str(row.get("merged_path", "")).strip()

    candidates: List[Path] = []
    if label and merged_filename:
        candidates.append(merged_root / "videos" / label / merged_filename)
    if merged_path:
        p = Path(merged_path)
        candidates.append(p if p.is_absolute() else merged_root / "videos" / p)

    video_path: Optional[Path] = None
    for candidate in candidates:
        if candidate.exists():
            video_path = candidate
            break

    if video_path is None:
        return {
            "status": "missing",
            "row_index": int(row_index),
            "merged_filename": merged_filename,
            "split": str(row.get("split", "")),
            "label": str(row.get("label", "")),
            "records": [],
        }

    label_int, label_name = normalize_label(row.get("label", ""))
    split = str(row.get("split", "unknown"))
    source_dataset = str(row.get("source_dataset", "unknown"))
    video_id = stable_video_id(row, row_index, video_path)

    # Check if we can use the new JRTIP full pipeline.
    use_full = bool(args.use_full_pipeline)

    records: List[Dict[str, Any]] = []
    gate_params: Dict[str, Any] = {}
    preprocess_status = "success"
    detector_model = "fallback_wholeframe"
    stacks_used: List[int] = []
    fallback_reason: Optional[str] = None

    if args.dry_run:
        # Dry run: just create placeholder records
        for clip_length in args.clip_lengths:
            prof = profile_name(args.preprocess_type, clip_length, args.size)
            cache_path = args.output_root / prof / "samples" / f"{video_id}_t{clip_length}.pt"
            records.append({
                "cache_path": str(cache_path.relative_to(args.output_root)).replace("\\", "/"),
                "video_id": video_id,
                "label": label_int,
                "label_name": label_name,
                "split": split,
                "source_dataset": source_dataset,
                "source_video": str(video_path),
                "original_path": str(row.get("original_path", "")),
                "merged_filename": merged_filename,
                "clip_length": int(clip_length),
                "height": int(args.size),
                "width": int(args.size),
                "dtype": "uint8",
                "preprocess_type": args.preprocess_type,
                "preprocess_status": "dry_run",
                "detector_model": detector_model,
                "gate_params": json.dumps(gate_params, separators=(",", ":")),
                "frame_count": 0,
                "sample_fps": float(args.sample_fps),
                "sha256": "dry_run",
                "stacks_used": "",
            })
        return {"status": "ok", "row_index": int(row_index), "records": records}

    try:
        stacks = []
        full_frames_rgb: Optional[np.ndarray] = None
        if use_full:
            try:
                full_frames_rgb, gate_params, preprocess_status = process_video_jrtip_kinematic(video_path, args)
                detector_model = "yolo11n_bytetrack_hdbscan_kinematic_gate"
                if preprocess_status != "success":
                    fallback_reason = str(gate_params.get("fallback_reason", preprocess_status))
            except Exception as e:
                log.warning("JRTIP kinematic pipeline failed for %s: %s. Falling back to whole-frame.", video_path, e)
                fallback_reason = f"pipeline_error:{type(e).__name__}"
                use_full = False

        if use_full:
            if full_frames_rgb is None or len(full_frames_rgb) == 0:
                fallback_reason = "no_kinematic_frames"
                use_full = False

        if not use_full:
            # Fallback: whole-frame decode
            max_len = max(args.clip_lengths)
            frames_rgb = decode_frames_ffmpeg(
                video_path, args.ffmpeg_bin, args.size, max_len, args.sample_fps
            )
            # frames_rgb is (T, H, W, 3) uint8
            # Letterbox to ensure square (though size already 224x224 from ffmpeg)
            if frames_rgb.shape[1] != args.size or frames_rgb.shape[2] != args.size:
                frames_rgb = letterbox_frames(frames_rgb, args.size)

            gate_params = {
                "method": "fallback_wholeframe",
                "sample_fps": float(args.sample_fps),
                "fallback_reason": fallback_reason or ("missing_dependencies" if not args.force_fallback else "forced_fallback"),
            }
            detector_model = "fallback_wholeframe"
            preprocess_status = "fallback_wholeframe"
            all_frames = frames_rgb
            actual_frame_count = all_frames.shape[0]
            # stacks already empty

        # Generate clips for each clip length
        for clip_length in args.clip_lengths:
            prof = profile_name(args.preprocess_type, clip_length, args.size)
            profile_dir = args.output_root / prof
            samples_dir = profile_dir / "samples"
            cache_path = samples_dir / f"{video_id}_t{clip_length}.pt"
            rel_cache_path = cache_path.relative_to(args.output_root)

            frames_tensor: Optional[torch.Tensor] = None
            stack_label = ""
            actual_frame_count = 0
            stacks_used = []

            if use_full and full_frames_rgb is not None:
                all_frames = full_frames_rgb
                actual_frame_count = int(all_frames.shape[0])
                frames_tensor = make_clip_tensor(all_frames, clip_length)
                stacks_used = []

            else:
                # Fallback: use directly decoded frames
                all_frames = frames_rgb  # (T, H, W, 3)
                actual_frame_count = all_frames.shape[0]
                frames_tensor = make_clip_tensor(all_frames, clip_length)
                # Label comes from split
                stacks_used = []

            # Save tensor
            if not args.dry_run:
                samples_dir.mkdir(parents=True, exist_ok=True)
                if cache_path.exists() and not args.overwrite:
                    # Use existing file
                    pass
                else:
                    # Save with metadata
                    payload = {
                        "video": frames_tensor,
                        "label": label_int,
                        "label_name": label_name,
                        "stack_label": stack_label,
                        "split": split,
                        "source_dataset": source_dataset,
                        "source_video": str(video_path),
                        "original_path": str(row.get("original_path", "")),
                        "merged_filename": merged_filename,
                        "video_id": video_id,
                        "clip_length": int(clip_length),
                        "height": int(args.size),
                        "width": int(args.size),
                        "sample_fps": float(args.sample_fps),
                        "preprocess_type": args.preprocess_type,
                        "dtype": "uint8",
                        "preprocess_status": preprocess_status,
                        "detector_model": detector_model,
                        "gate_params": gate_params,
                        "stacks_used": stacks_used,
                    }
                    torch.save(payload, cache_path)

                sha = file_sha256(cache_path) if cache_path.exists() else "missing"

                records.append({
                    "cache_path": str(rel_cache_path).replace("\\", "/"),
                    "video_id": video_id,
                    "label": label_int,
                    "label_name": label_name,
                    "stack_label": stack_label,
                    "split": split,
                    "source_dataset": source_dataset,
                    "source_video": str(video_path),
                    "original_path": str(row.get("original_path", "")),
                    "merged_filename": merged_filename,
                    "clip_length": int(clip_length),
                    "height": int(args.size),
                    "width": int(args.size),
                    "dtype": "uint8",
                    "preprocess_type": args.preprocess_type,
                    "preprocess_status": preprocess_status,
                    "detector_model": detector_model,
                    "gate_params": json.dumps(gate_params, separators=(",", ":")),
                    "frame_count": int(actual_frame_count),
                    "sample_fps": float(args.sample_fps),
                    "sha256": sha,
                    "stacks_used": ";".join(str(s) for s in stacks_used) if stacks_used else "",
                })
            else:
                records.append({
                    "cache_path": str(rel_cache_path).replace("\\", "/"),
                    "video_id": video_id,
                    "label": label_int,
                    "label_name": label_name,
                    "stack_label": stack_label,
                    "split": split,
                    "source_dataset": source_dataset,
                    "source_video": str(video_path),
                    "original_path": str(row.get("original_path", "")),
                    "merged_filename": merged_filename,
                    "clip_length": int(clip_length),
                    "height": int(args.size),
                    "width": int(args.size),
                    "dtype": "uint8",
                    "preprocess_type": args.preprocess_type,
                    "preprocess_status": preprocess_status,
                    "detector_model": detector_model,
                    "gate_params": json.dumps(gate_params, separators=(",", ":")),
                    "frame_count": int(actual_frame_count),
                    "sample_fps": float(args.sample_fps),
                    "sha256": "dry_run",
                    "stacks_used": ";".join(str(s) for s in stacks_used) if stacks_used else "",
                })

        return {"status": "ok", "row_index": int(row_index), "records": records}

    except Exception as exc:
        log.exception("Failed processing %s: %s", video_path, exc)
        return {
            "status": "failed",
            "row_index": int(row_index),
            "video_path": str(video_path),
            "error": str(exc),
            "records": [],
        }


def summarize(
    records: List[Dict[str, Any]],
    statuses: List[Dict[str, Any]],
    args: argparse.Namespace,
    pipeline_used: bool,
) -> Dict[str, Any]:
    """Build stats dictionary from manifest records."""
    stats: Dict[str, Any] = {
        "output_root": str(args.output_root),
        "preprocess_type": args.preprocess_type,
        "clip_lengths": [int(x) for x in args.clip_lengths],
        "size": int(args.size),
        "sample_fps": float(args.sample_fps),
        "dry_run": bool(args.dry_run),
        "full_pipeline_used": pipeline_used,
        "total_rows": int(len(statuses)),
        "processed_videos": int(sum(1 for s in statuses if s["status"] == "ok")),
        "missing_videos": int(sum(1 for s in statuses if s["status"] == "missing")),
        "failed_videos": int(sum(1 for s in statuses if s["status"] == "failed")),
        "profiles": {},
        "failures": [s for s in statuses if s["status"] != "ok"][:200],
    }

    for clip_length in args.clip_lengths:
        prof = profile_name(args.preprocess_type, clip_length, args.size)
        sub = [r for r in records if int(r["clip_length"]) == int(clip_length)]
        stats["profiles"][prof] = {
            "samples": int(len(sub)),
            "by_split": dict(pd.Series([r["split"] for r in sub]).value_counts()) if sub else {},
            "by_label": dict(pd.Series([r["label_name"] for r in sub]).value_counts()) if sub else {},
            "by_preprocess_status": dict(pd.Series([r.get("preprocess_status", "unknown") for r in sub]).value_counts()) if sub else {},
        }

    return stats


def write_outputs(
    records: List[Dict[str, Any]],
    stats: Dict[str, Any],
    args: argparse.Namespace,
) -> None:
    """Write manifest.csv in each profile plus root stats.json."""
    if args.dry_run:
        log.info("Dry-run enabled; no manifests or tensors written.")
        return

    args.output_root.mkdir(parents=True, exist_ok=True)

    # Manifest fields - include all required plus extra metadata
    fields = [
        "cache_path",
        "video_id",
        "label",
        "label_name",
        "stack_label",
        "split",
        "source_dataset",
        "source_video",
        "original_path",
        "merged_filename",
        "clip_length",
        "height",
        "width",
        "dtype",
        "preprocess_type",
        "preprocess_status",
        "detector_model",
        "gate_params",
        "frame_count",
        "sample_fps",
        "sha256",
        "stacks_used",
    ]

    for clip_length in args.clip_lengths:
        prof = profile_name(args.preprocess_type, clip_length, args.size)
        profile_dir = args.output_root / prof
        profile_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = profile_dir / "manifest.csv"
        sub = [r for r in records if int(r["clip_length"]) == int(clip_length)]

        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(sub)
        log.info("Wrote %s (%d rows)", manifest_path, len(sub))

    stats_path = args.output_root / "movinet_preprocess_stats.json"
    stats_path.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )
    log.info("Wrote %s", stats_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build MoViNet preprocessed cache from JRTIP pipeline. "
                    "Outputs to result/gpu_cache/ by default."
    )
    parser.add_argument("--splits-csv", type=Path, default=Path("data/splits/splits.csv"))
    parser.add_argument("--merged-root", type=Path, default=Path("data/merged"))
    parser.add_argument("--output-root", type=Path, default=Path("result/gpu_cache"))
    parser.add_argument("--clip-lengths", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--size", type=int, default=224, help="Output frame size (H=W)")
    parser.add_argument("--sample-fps", type=float, default=8.0,
                        help="Frame sampling rate for fallback and FFmpeg decode")
    parser.add_argument("--preprocess-type", default="movinet_preprocessed",
                        help="Profile prefix in output directory name")
    parser.add_argument("--ffmpeg-bin", default=None, help="Path to ffmpeg binary")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--executor", choices=["auto", "thread", "process"], default="auto",
                        help="Parallel executor. auto uses process for full pipeline and thread for fallback.")
    parser.add_argument("--opencv-threads", type=int, default=1,
                        help="OpenCV CPU threads per worker process.")
    parser.add_argument("--torch-threads", type=int, default=1,
                        help="Torch CPU threads per worker process.")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N videos")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing cache files")
    parser.add_argument("--dry-run", action="store_true", help="Check inputs but don't write output")
    parser.add_argument("--person-model", default="yolo11n.pt",
                        help="Ultralytics YOLO person detector. Default uses YOLOv11n PyTorch weights.")
    parser.add_argument("--detector-device", default="0",
                        help="Ultralytics device for YOLOv11n, e.g. 0 or cpu.")
    parser.add_argument("--detector-imgsz", type=int, default=640)
    parser.add_argument("--half", action="store_true",
                        help="Use FP16 detector inference on CUDA devices.")
    parser.add_argument("--person-conf", type=float, default=0.25)
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--cluster-min-pts", type=int, default=2)
    parser.add_argument(
        "--hdbscan-epsilon",
        type=float,
        default=0.0,
        help="HDBSCAN cluster_selection_epsilon in pixels; larger values merge nearby person clusters more easily.",
    )
    parser.add_argument("--iou-gate", type=float, default=0.85)
    parser.add_argument("--velocity-gate", type=float, default=0.05)
    parser.add_argument("--kappa-frames", type=int, default=2)
    parser.add_argument(
        "--person-match-dist",
        type=float,
        default=1.5,
        help="Maximum normalized center distance for matching person boxes between adjacent frames when YOLO IDs are unavailable.",
    )
    parser.add_argument("--crowd-retain-frames", type=int, default=3,
                        help="Keep disappeared crowd IDs for N sampled frames, matching C++ retention.")
    parser.add_argument("--crop-scale", type=float, default=1.5)
    parser.add_argument("--force-fallback", action="store_true",
                        help="Force use of fallback whole-frame method even if full pipeline is available")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_result_path(args.output_root, "output_root")
    reject_legacy_path(args.splits_csv, "splits_csv")
    reject_legacy_path(args.merged_root, "merged_root")
    cv2.setNumThreads(max(0, int(args.opencv_threads)))
    torch.set_num_threads(max(1, int(args.torch_threads)))

    # Validate inputs
    if not args.splits_csv.exists():
        raise FileNotFoundError(f"Splits CSV not found: {args.splits_csv}")
    if not args.merged_root.exists():
        raise FileNotFoundError(f"Merged root not found: {args.merged_root}")

    # Resolve ffmpeg
    if not args.dry_run:
        import shutil
        args.ffmpeg_bin = find_ffmpeg_bin(args.ffmpeg_bin)
    else:
        args.ffmpeg_bin = "ffmpeg"  # dummy for dry-run

    # Sort and deduplicate clip lengths
    args.clip_lengths = sorted(set(int(x) for x in args.clip_lengths))

    # New JRTIP preprocess does not use fightOD/config.yaml. It uses YOLOv11n
    # person detection + ByteTrack + HDBSCAN + kinematic gate.
    pipeline_available = False
    config = None
    if not args.force_fallback:
        missing = []
        if YOLO is None:
            missing.append("ultralytics")
        if hdbscan is None:
            missing.append("hdbscan")
        if missing:
            log.warning("Missing dependencies for JRTIP kinematic pipeline: %s. Using fallback.", ", ".join(missing))
        else:
            pipeline_available = True

    args.use_full_pipeline = pipeline_available and not args.force_fallback

    if args.use_full_pipeline:
        log.info(
            "Using JRTIP YOLOv11n/ByteTrack/HDBSCAN/Kinematic-Gate pipeline "
            "(person_model=%s, device=%s)",
            args.person_model,
            args.detector_device,
        )
    else:
        log.info("Using fallback whole-frame method")

    # Read splits
    df = pd.read_csv(args.splits_csv)
    if args.limit is not None:
        df = df.head(args.limit).copy()

    rows = [(int(idx), row.to_dict()) for idx, row in df.iterrows()]
    log.info(
        "Processing %d rows; output=%s; full_pipeline=%s",
        len(rows), args.output_root, args.use_full_pipeline
    )

    # Process videos
    records: List[Dict[str, Any]] = []
    statuses: List[Dict[str, Any]] = []
    workers = max(1, int(args.workers))
    if args.executor == "auto":
        executor_kind = "process" if args.use_full_pipeline and workers > 1 else "thread"
    else:
        executor_kind = args.executor
    executor_cls = ProcessPoolExecutor if executor_kind == "process" else ThreadPoolExecutor
    start_time = time.time()

    log.info(
        "Executor=%s workers=%d half=%s opencv_threads=%d torch_threads=%d",
        executor_kind,
        workers,
        bool(args.half),
        int(args.opencv_threads),
        int(args.torch_threads),
    )

    with executor_cls(max_workers=workers) as pool:
        futures = [
            pool.submit(process_one, idx, row, args, config if args.use_full_pipeline else None)
            for idx, row in rows
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing videos"):
            status = future.result()
            statuses.append(status)
            records.extend(status.get("records", []))

    # Summarize and write outputs
    stats = summarize(records, statuses, args, args.use_full_pipeline)
    write_outputs(records, stats, args)

    log.info(
        "Done. processed=%d missing=%d failed=%d samples=%d elapsed=%.1fs videos/s=%.2f",
        stats["processed_videos"],
        stats["missing_videos"],
        stats["failed_videos"],
        len(records),
        time.time() - start_time,
        float(len(rows)) / max(time.time() - start_time, 1e-6),
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log.error("%s", exc)
        sys.exit(1)
