"""Direct video datasets for training from data/merged without cache files."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate

try:
    import cv2
except Exception as exc:  # pragma: no cover
    cv2 = None
    _CV2_IMPORT_ERROR = exc
else:
    _CV2_IMPORT_ERROR = None


log = logging.getLogger("direct_video_dataset")

LABEL_TO_INT = {"non_violence": 0, "violence": 1, "0": 0, "1": 1}
INT_TO_LABEL = {0: "non_violence", 1: "violence"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".mpeg", ".mpg"}
_PERSON_MODEL = None


@dataclass
class DirectVideoMetadata:
    video_id: str
    source_video: str
    split: str
    label_name: str
    source_dataset: str
    original_path: str
    merged_filename: str
    clip_length: int
    height: int
    width: int
    preprocess_type: str
    frame_count: int
    decode_time_ms: float


def normalize_label(raw_label: Any) -> tuple[int, str]:
    label_name = str(raw_label).strip()
    if label_name not in LABEL_TO_INT:
        raise ValueError(f"Unsupported label '{raw_label}'. Expected violence/non_violence or 0/1.")
    label_int = int(LABEL_TO_INT[label_name])
    return label_int, INT_TO_LABEL[label_int]


def resolve_video_path(row: pd.Series, merged_root: str | Path) -> Path:
    root = Path(merged_root)
    label = str(row.get("label", "")).strip()
    split = str(row.get("split", "")).strip()
    merged_filename = str(row.get("merged_filename", "")).strip()
    filename = str(row.get("filename", "")).strip()
    merged_path = str(row.get("merged_path", "")).strip()

    candidates: list[Path] = []
    for name in (merged_filename, filename):
        if not name:
            continue
        if split and label:
            candidates.append(root / f"{split}_videos" / label / name)
        if label:
            candidates.append(root / "videos" / label / name)
        candidates.append(root / name)

    if merged_path:
        path = Path(merged_path)
        candidates.append(path if path.is_absolute() else root / path)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = ", ".join(str(x) for x in candidates[:5])
    raise FileNotFoundError(f"Video not found for {merged_filename or filename}. Checked: {searched}")


def make_video_id(row: pd.Series, idx: int, video_path: Path) -> str:
    source = str(row.get("source_dataset", "unknown")).replace(" ", "_")
    split = str(row.get("split", "unknown"))
    return f"{source}_{split}_{video_path.stem}_{idx}"


def _resize_center_crop(frame_rgb: np.ndarray, size: int) -> np.ndarray:
    height, width = frame_rgb.shape[:2]
    if height <= 0 or width <= 0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    scale = max(float(size) / float(width), float(size) / float(height))
    new_w = max(size, int(round(width * scale)))
    new_h = max(size, int(round(height * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(frame_rgb, (new_w, new_h), interpolation=interp)
    x0 = max(0, (new_w - size) // 2)
    y0 = max(0, (new_h - size) // 2)
    return resized[y0:y0 + size, x0:x0 + size].copy()


def _uniform_indices(total_frames: int, clip_length: int) -> np.ndarray:
    if total_frames <= 0:
        return np.zeros((clip_length,), dtype=np.int64)
    return np.linspace(0, total_frames - 1, int(clip_length)).round().astype(np.int64)


def find_ffmpeg_bin(ffmpeg_bin: str | Path | None = None) -> str:
    candidate = str(ffmpeg_bin or os.getenv("FFMPEG_BIN") or shutil.which("ffmpeg") or "ffmpeg")
    resolved = shutil.which(candidate) if candidate == "ffmpeg" else candidate
    if not resolved or (candidate != "ffmpeg" and not Path(resolved).exists()):
        raise FileNotFoundError(
            "ffmpeg not found. Add ffmpeg to PATH, set FFMPEG_BIN, or pass --ffmpeg-bin."
        )
    return resolved


def decode_video_ffmpeg(
    video_path: Path,
    clip_length: int,
    size: int,
    ffmpeg_bin: str | Path | None = None,
) -> tuple[np.ndarray, int]:
    """Decode a video through FFmpeg, center-crop to square, then sample/pad to clip_length."""
    ffmpeg = find_ffmpeg_bin(ffmpeg_bin)
    frame_bytes = int(size) * int(size) * 3
    vf = f"scale={size}:{size}:force_original_aspect_ratio=increase,crop={size}:{size},format=rgb24"
    cmd = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-an",
        "-sn",
        "-vf",
        vf,
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
    if not proc.stdout:
        raise RuntimeError(f"ffmpeg decoded zero frames for {video_path}")
    usable = (len(proc.stdout) // frame_bytes) * frame_bytes
    if usable <= 0:
        raise RuntimeError(f"ffmpeg output is smaller than one RGB frame for {video_path}")
    frames = np.frombuffer(proc.stdout[:usable], dtype=np.uint8).reshape(-1, int(size), int(size), 3)
    frame_count = int(frames.shape[0])
    return sample_or_pad_frames(frames, int(clip_length)).copy(), frame_count


def decode_video_opencv(video_path: Path, clip_length: int, size: int) -> tuple[np.ndarray, int]:
    if cv2 is None:
        raise RuntimeError("opencv-python/cv2 is required for direct video training.") from _CV2_IMPORT_ERROR

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        indices = _uniform_indices(total_frames, clip_length)
        frames: list[np.ndarray] = []
        last_frame: Optional[np.ndarray] = None

        for frame_idx in indices:
            if total_frames > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                if last_frame is not None:
                    frames.append(last_frame.copy())
                    continue
                frames.append(np.zeros((size, size, 3), dtype=np.uint8))
                continue
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb = _resize_center_crop(frame_rgb, size)
            last_frame = frame_rgb
            frames.append(frame_rgb)
    finally:
        cap.release()

    if not frames:
        frames = [np.zeros((size, size, 3), dtype=np.uint8) for _ in range(int(clip_length))]
    return np.stack(frames, axis=0), int(total_frames)


def decode_video_uniform(
    video_path: Path,
    clip_length: int,
    size: int,
    video_decoder: str = "ffmpeg",
    ffmpeg_bin: str | Path | None = None,
) -> tuple[np.ndarray, int]:
    decoder = str(video_decoder).lower().strip()
    if decoder == "ffmpeg":
        return decode_video_ffmpeg(video_path, clip_length, size, ffmpeg_bin=ffmpeg_bin)
    if decoder in {"opencv", "cv2"}:
        return decode_video_opencv(video_path, clip_length, size)
    raise ValueError(f"Unsupported video_decoder={video_decoder!r}. Expected 'ffmpeg' or 'opencv'.")


def _expand_xyxy(box: np.ndarray, scale: float, width: int, height: int) -> np.ndarray:
    x1, y1, x2, y2 = [float(x) for x in box]
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    bw = max(1.0, (x2 - x1) * float(scale))
    bh = max(1.0, (y2 - y1) * float(scale))
    return np.array(
        [
            max(0.0, cx - bw * 0.5),
            max(0.0, cy - bh * 0.5),
            min(float(width), cx + bw * 0.5),
            min(float(height), cy + bh * 0.5),
        ],
        dtype=np.float32,
    )


def _crop_resize_xyxy(frame_rgb: np.ndarray, box: np.ndarray, size: int) -> np.ndarray:
    height, width = frame_rgb.shape[:2]
    x1, y1, x2, y2 = _expand_xyxy(box, 1.5, width, height).round().astype(int).tolist()
    if x2 <= x1 or y2 <= y1:
        return _resize_center_crop(frame_rgb, size)
    crop = frame_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return _resize_center_crop(frame_rgb, size)
    interp = cv2.INTER_AREA if crop.shape[0] > size or crop.shape[1] > size else cv2.INTER_LINEAR
    return cv2.resize(crop, (size, size), interpolation=interp)


def get_person_model(model_path: str):
    global _PERSON_MODEL
    if _PERSON_MODEL is None:
        from ultralytics import YOLO  # type: ignore

        _PERSON_MODEL = YOLO(model_path, task="detect")
    return _PERSON_MODEL


def sample_or_pad_frames(frames: np.ndarray, clip_length: int) -> np.ndarray:
    total = int(frames.shape[0])
    if total <= 0:
        raise ValueError("Cannot sample empty frame array.")
    if total >= int(clip_length):
        idx = np.linspace(0, total - 1, int(clip_length)).round().astype(np.int64)
        return frames[idx]
    pad = np.repeat(frames[-1:, :, :, :], int(clip_length) - total, axis=0)
    return np.concatenate([frames, pad], axis=0)


def frames_to_tensor(frames: np.ndarray, normalize: bool) -> torch.Tensor:
    clip = np.ascontiguousarray(np.transpose(frames, (3, 0, 1, 2)))
    video = torch.from_numpy(clip).float().div_(255.0)
    if normalize:
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1, 1)
        video = (video - mean) / std
    return video.contiguous()


def _standardize_clip(video: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = video.mean()
    std = video.std()
    if float(std) < eps:
        return video - mean
    return (video - mean) / std


def _uniform_temporal_sample(video: torch.Tensor, frames: int) -> torch.Tensor:
    current = int(video.shape[1])
    frames = int(frames)
    if current == frames:
        return video
    if current > frames:
        idx = torch.linspace(0, current - 1, frames).round().long()
        return video[:, idx, :, :]
    pad = video[:, -1:, :, :].repeat(1, frames - current, 1, 1)
    return torch.cat([video, pad], dim=1)


def _resize_clip(video: torch.Tensor, output_size: Optional[int]) -> torch.Tensor:
    if output_size is None or int(video.shape[-1]) == int(output_size):
        return video
    x = video.unsqueeze(0)
    x = F.interpolate(
        x,
        size=(int(video.shape[1]), int(output_size), int(output_size)),
        mode="trilinear",
        align_corners=False,
    )
    return x.squeeze(0)


def _farneback_flow_from_rgb(video: torch.Tensor) -> torch.Tensor:
    if cv2 is None:
        raise RuntimeError("JOSENet direct training requires opencv-python/cv2.") from _CV2_IMPORT_ERROR
    rgb = video.detach().cpu().clamp(0, 1).permute(1, 2, 3, 0).numpy()
    rgb_u8 = (rgb * 255.0).round().astype(np.uint8)
    flows: list[np.ndarray] = []
    for idx in range(max(0, rgb_u8.shape[0] - 1)):
        prev_gray = cv2.cvtColor(rgb_u8[idx], cv2.COLOR_RGB2GRAY)
        next_gray = cv2.cvtColor(rgb_u8[idx + 1], cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flows.append(flow.astype(np.float32))
    flows.append(np.zeros((*rgb_u8.shape[1:3], 2), dtype=np.float32))
    return _standardize_clip(torch.from_numpy(np.stack(flows, axis=0)).permute(3, 0, 1, 2).contiguous())


class DirectVideoDataset(Dataset):
    """Read clips directly from data/merged according to data/splits/splits.csv."""

    def __init__(
        self,
        splits_csv: str | Path = "data/splits/splits.csv",
        merged_root: str | Path = "data/merged",
        split: str = "train",
        clip_length: int = 16,
        size: int = 224,
        preprocess_type: str = "wholeframe",
        normalize: bool = True,
        enable_movinet_pipeline: bool = True,
        movinet_person_model: str | Path = "yolo11n.pt",
        detector_device: str = "0",
        video_decoder: str = "ffmpeg",
        ffmpeg_bin: str | Path | None = None,
    ) -> None:
        self.splits_csv = Path(splits_csv)
        self.merged_root = Path(merged_root)
        self.split = split
        self.clip_length = int(clip_length)
        self.size = int(size)
        self.preprocess_type = preprocess_type
        self.normalize = bool(normalize)
        self.enable_movinet_pipeline = bool(enable_movinet_pipeline)
        self.movinet_person_model = str(movinet_person_model)
        self.detector_device = str(detector_device)
        self.video_decoder = str(video_decoder)
        self.ffmpeg_bin = str(ffmpeg_bin) if ffmpeg_bin is not None else None

        if not self.splits_csv.exists():
            raise FileNotFoundError(f"Splits CSV not found: {self.splits_csv}")
        if not self.merged_root.exists():
            raise FileNotFoundError(f"Merged root not found: {self.merged_root}")

        df = pd.read_csv(self.splits_csv)
        if "split" not in df.columns:
            raise ValueError(f"Split column missing from {self.splits_csv}")
        self.rows = df[df["split"].astype(str) == str(split)].reset_index(drop=False)
        if self.rows.empty:
            raise ValueError(f"No rows for split={split} in {self.splits_csv}")
        self.manifest = self.rows.copy()
        self.manifest["label"] = [normalize_label(x)[0] for x in self.manifest["label"].tolist()]

    def __len__(self) -> int:
        return int(len(self.rows))

    def _decode_movinet_preprocessed(self, video_path: Path) -> tuple[np.ndarray, int, str]:
        if not self.enable_movinet_pipeline:
            frames, total = decode_video_uniform(
                video_path,
                self.clip_length,
                self.size,
                video_decoder=self.video_decoder,
                ffmpeg_bin=self.ffmpeg_bin,
            )
            return frames, total, "wholeframe_fallback_disabled"
        try:
            model = get_person_model(self.movinet_person_model)
        except Exception as exc:
            log.warning("MoViNet direct preprocess unavailable, using whole-frame fallback: %s", exc)
            frames, total = decode_video_uniform(
                video_path,
                self.clip_length,
                self.size,
                video_decoder=self.video_decoder,
                ffmpeg_bin=self.ffmpeg_bin,
            )
            return frames, total, "wholeframe_fallback_import"

        if cv2 is None:
            raise RuntimeError("opencv-python/cv2 is required for direct MoViNet preprocessing.") from _CV2_IMPORT_ERROR

        try:
            decoded_frames, total = decode_video_uniform(
                video_path,
                self.clip_length,
                self.size,
                video_decoder=self.video_decoder,
                ffmpeg_bin=self.ffmpeg_bin,
            )
            frames: list[np.ndarray] = []
            used_person_crop = 0
            for frame_rgb in decoded_frames:
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                result = model.predict(
                    source=frame_bgr,
                    conf=0.25,
                    classes=[0],
                    device=self.detector_device,
                    verbose=False,
                    imgsz=640,
                )[0]
                boxes = None
                if result.boxes is not None and result.boxes.xyxy is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
                if boxes is not None and len(boxes) > 0:
                    union = np.array(
                        [
                            boxes[:, 0].min(),
                            boxes[:, 1].min(),
                            boxes[:, 2].max(),
                            boxes[:, 3].max(),
                        ],
                        dtype=np.float32,
                    )
                    frames.append(_crop_resize_xyxy(frame_rgb, union, self.size))
                    used_person_crop += 1
                else:
                    frames.append(frame_rgb.copy())
            status = "person_crop" if used_person_crop > 0 else "wholeframe_no_person"
            return np.stack(frames, axis=0), int(total), status
        except Exception as exc:
            log.warning("MoViNet direct preprocess failed for %s, using whole-frame fallback: %s", video_path, exc)
            frames, total = decode_video_uniform(
                video_path,
                self.clip_length,
                self.size,
                video_decoder=self.video_decoder,
                ffmpeg_bin=self.ffmpeg_bin,
            )
            return frames, total, "wholeframe_fallback_error"

    def __getitem__(self, idx: int):
        start = time.perf_counter()
        row = self.rows.iloc[int(idx)]
        video_path = resolve_video_path(row, self.merged_root)
        label_int, label_name = normalize_label(row.get("label", ""))

        preprocess_status = self.preprocess_type
        if self.preprocess_type == "movinet_preprocessed":
            frames, frame_count, preprocess_status = self._decode_movinet_preprocessed(video_path)
        else:
            frames, frame_count = decode_video_uniform(
                video_path,
                self.clip_length,
                self.size,
                video_decoder=self.video_decoder,
                ffmpeg_bin=self.ffmpeg_bin,
            )

        video = frames_to_tensor(frames, normalize=self.normalize)
        decode_time_ms = (time.perf_counter() - start) * 1000.0
        metadata = DirectVideoMetadata(
            video_id=make_video_id(row, int(row.get("index", idx)), video_path),
            source_video=str(video_path),
            split=str(row.get("split", self.split)),
            label_name=label_name,
            source_dataset=str(row.get("source_dataset", "unknown")),
            original_path=str(row.get("original_path", "")),
            merged_filename=str(row.get("merged_filename", video_path.name)),
            clip_length=self.clip_length,
            height=self.size,
            width=self.size,
            preprocess_type=preprocess_status,
            frame_count=int(frame_count),
            decode_time_ms=float(decode_time_ms),
        )
        return video, int(label_int), metadata

    def get_class_distribution(self) -> dict[int, int]:
        labels = [normalize_label(x)[0] for x in self.rows["label"].tolist()]
        return {int(k): int(v) for k, v in pd.Series(labels).value_counts().to_dict().items()}


class DirectModelDataset(Dataset):
    def __init__(self, base: DirectVideoDataset, output_size: Optional[int] = None) -> None:
        self.base = base
        self.output_size = output_size

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        video, label, metadata = self.base[idx]
        video = _resize_clip(video, self.output_size)
        return video.contiguous(), int(label), metadata


class DirectSlowFastDataset(Dataset):
    def __init__(self, base: DirectVideoDataset, slow_frames: int = 8, fast_frames: int = 32) -> None:
        self.base = base
        self.slow_frames = int(slow_frames)
        self.fast_frames = int(fast_frames)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        video, label, metadata = self.base[idx]
        fast = _uniform_temporal_sample(video, self.fast_frames)
        slow = _uniform_temporal_sample(fast, self.slow_frames)
        return (slow.contiguous(), fast.contiguous()), int(label), metadata


class DirectJOSENetDataset(Dataset):
    effective_frames = 16

    def __init__(self, base: DirectVideoDataset) -> None:
        if int(base.clip_length) != self.effective_frames:
            raise ValueError("JOSENet official architecture supports only 16-frame clips.")
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        video, label, metadata = self.base[idx]
        video = _uniform_temporal_sample(video, self.effective_frames)
        # The base JOSENet stream must be unnormalized RGB in [0, 1].
        if self.base.normalize:
            raise ValueError("DirectJOSENetDataset requires base.normalize=False.")
        flow = _farneback_flow_from_rgb(video)
        rgb = _standardize_clip(video)
        return (rgb.contiguous(), flow.contiguous()), int(label), metadata


def collate_direct(batch):
    if batch and isinstance(batch[0], tuple) and len(batch[0]) == 3:
        inputs = [item[0] for item in batch]
        labels = torch.tensor([int(item[1]) for item in batch], dtype=torch.long)
        metadata = [item[2] for item in batch]
        return default_collate(inputs), labels, metadata
    return default_collate(batch)


def build_direct_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    kwargs = {
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "drop_last": bool(drop_last),
        "collate_fn": collate_direct,
    }
    if int(num_workers) > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def make_direct_model_dataset(
    model_name: str,
    split: str,
    splits_csv: str | Path = "data/splits/splits.csv",
    merged_root: str | Path = "data/merged",
    clip_length: int = 16,
    normalize: bool = True,
    video_decoder: str = "ffmpeg",
    ffmpeg_bin: str | Path | None = None,
):
    key = model_name.lower().replace("-", "_")
    if key == "slowfast" and int(clip_length) < 32:
        raise ValueError("SlowFast direct adapter requires clip_length 32 or 64.")
    if key == "josenet" and int(clip_length) != 16:
        raise ValueError("JOSENet direct adapter supports only 16-frame clips.")
    if key == "c3d" and int(clip_length) != 16:
        raise ValueError("C3D direct adapter supports only 16-frame clips.")
    if key not in {"c3d", "i3d", "resnet_lstm", "swin3d", "josenet", "slowfast"}:
        raise ValueError(f"Unsupported model_name: {model_name}")

    base = DirectVideoDataset(
        splits_csv=splits_csv,
        merged_root=merged_root,
        split=split,
        clip_length=int(clip_length),
        size=224,
        preprocess_type="wholeframe",
        normalize=False if key == "josenet" else normalize,
        video_decoder=video_decoder,
        ffmpeg_bin=ffmpeg_bin,
    )

    if key == "slowfast":
        return DirectSlowFastDataset(base)
    if key == "josenet":
        return DirectJOSENetDataset(base)
    return DirectModelDataset(base, output_size=112 if key == "c3d" else 224)


def measure_decode_throughput(dataset: Dataset, max_samples: int = 16) -> dict[str, float | int]:
    samples = min(int(max_samples), len(dataset))
    if samples <= 0:
        return {"samples": 0, "clips_per_sec": 0.0, "mean_decode_ms_per_clip": 0.0, "wall_time_sec": 0.0}
    decode_times: list[float] = []
    start = time.perf_counter()
    for idx in range(samples):
        item = dataset[idx]
        metadata = item[2] if isinstance(item, tuple) and len(item) >= 3 else None
        if metadata is not None and hasattr(metadata, "decode_time_ms"):
            decode_times.append(float(metadata.decode_time_ms))
    wall_time = time.perf_counter() - start
    mean_ms = float(sum(decode_times) / len(decode_times)) if decode_times else float((wall_time / samples) * 1000.0)
    return {
        "samples": int(samples),
        "clips_per_sec": float(samples / max(wall_time, 1e-9)),
        "mean_decode_ms_per_clip": mean_ms,
        "wall_time_sec": float(wall_time),
    }
