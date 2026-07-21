"""
PyTorch Dataset for JRTIP GPU clip caches.

The loader reads cache manifests produced by data/scripts/build_gpu_clip_cache.py
or future preprocess builders. It returns float32 tensors ready for model input
without decoding the source video again.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset

log = logging.getLogger("gpu_clip_cache_dataset")

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


def assert_new_cache_path(path: str | Path, field_name: str = "path") -> None:
    """Reject legacy cache/data roots for new JRTIP experiments."""
    text = str(path)
    normalized = text.replace("\\", "/")
    for marker in LEGACY_CACHE_MARKERS:
        if marker.replace("\\", "/") in normalized:
            raise ValueError(
                f"{field_name} points to legacy cache/data '{marker}': {path}. "
                "New JRTIP experiments must use result/gpu_cache."
            )


@dataclass
class ClipMetadata:
    cache_path: str
    video_id: str
    label_name: str
    split: str
    source_dataset: str
    source_video: str
    original_path: str
    merged_filename: str
    clip_length: int
    height: int
    width: int
    dtype: str
    preprocess_type: str
    frame_count: int
    sample_fps: float
    sha256: str


def _profile_name(preprocess_type: str, clip_length: int, size: int = 224) -> str:
    if preprocess_type in {"wholeframe", "wholeframe_rgb"}:
        return f"wholeframe_rgb_t{clip_length}_{size}"
    return f"{preprocess_type}_t{clip_length}_{size}"


class GpuClipCacheDataset(Dataset):
    """
    Dataset for cached tensors.

    Parameters
    ----------
    manifest_path:
        Direct path to a manifest.csv. If provided, it takes precedence.
    cache_root:
        Root cache directory. Defaults to result/gpu_cache.
    profile:
        Cache profile directory name, e.g. wholeframe_rgb_t16_224.
    clip_length / preprocess_type / size:
        Used to derive profile if profile is not supplied.
    split:
        Optional train/val/test filter.
    normalize:
        Apply ImageNet normalization manually.
    device:
        Optional target device. For high-throughput training, prefer moving
        batches to device in the training loop instead of per item.
    transform:
        Optional callable applied after conversion to float32 [0,1].
    """

    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1, 1)
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1, 1)

    def __init__(
        self,
        manifest_path: Optional[str | Path] = None,
        cache_root: str | Path = "result/gpu_cache",
        profile: Optional[str] = None,
        clip_length: Optional[int] = None,
        preprocess_type: str = "wholeframe",
        size: int = 224,
        split: Optional[str] = None,
        normalize: bool = False,
        device: Optional[str | torch.device] = None,
        transform: Optional[Any] = None,
    ) -> None:
        super().__init__()
        assert_new_cache_path(cache_root, "cache_root")
        if manifest_path is not None:
            assert_new_cache_path(manifest_path, "manifest_path")
        self.cache_root = Path(cache_root)
        self.split = split
        self.normalize = normalize
        self.device = torch.device(device) if device is not None else None
        self.transform = transform

        if manifest_path is None:
            if profile is None:
                if clip_length is None:
                    raise ValueError("Provide manifest_path, profile, or clip_length.")
                profile = _profile_name(preprocess_type, int(clip_length), int(size))
            self.profile_dir = self.cache_root / profile
            manifest_path = self.profile_dir / "manifest.csv"
        else:
            manifest_path = Path(manifest_path)
            self.profile_dir = manifest_path.parent

        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
        self.manifest = pd.read_csv(self.manifest_path)
        if self.manifest.empty:
            raise ValueError(f"Manifest is empty: {self.manifest_path}")

        if split is not None:
            if "split" not in self.manifest.columns:
                raise ValueError(f"Manifest has no split column: {self.manifest_path}")
            available = sorted(str(x) for x in self.manifest["split"].dropna().unique())
            if split not in available:
                raise ValueError(f"Split '{split}' not found. Available: {available}")
            self.manifest = self.manifest[self.manifest["split"] == split].reset_index(drop=True)

        if self.manifest.empty:
            raise ValueError(f"No samples remain after filtering split={split}")

        log.info("Loaded %d cached clips from %s", len(self.manifest), self.manifest_path)

    def __len__(self) -> int:
        return int(len(self.manifest))

    def _resolve_cache_path(self, cache_path: str) -> Path:
        p = Path(cache_path)
        if p.is_absolute():
            return p
        root_candidate = self.cache_root / p
        if root_candidate.exists():
            return root_candidate
        profile_candidate = self.profile_dir / p
        if profile_candidate.exists():
            return profile_candidate
        return root_candidate

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, ClipMetadata]:
        row = self.manifest.iloc[int(idx)]
        cache_path = self._resolve_cache_path(str(row["cache_path"]))
        if not cache_path.exists():
            raise FileNotFoundError(f"Cached tensor missing: {cache_path}")

        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        video = payload["video"]
        if not torch.is_tensor(video):
            video = torch.as_tensor(video)
        if video.dtype != torch.uint8:
            video = video.to(torch.uint8)
        video = video.float().div_(255.0)

        if self.normalize:
            mean = self.IMAGENET_MEAN.to(video.device)
            std = self.IMAGENET_STD.to(video.device)
            video = (video - mean) / std

        if self.transform is not None:
            video = self.transform(video)

        if self.device is not None:
            video = video.to(self.device, non_blocking=True)

        label = int(row["label"])
        metadata = ClipMetadata(
            cache_path=str(row.get("cache_path", "")),
            video_id=str(row.get("video_id", "")),
            label_name=str(row.get("label_name", "")),
            split=str(row.get("split", "")),
            source_dataset=str(row.get("source_dataset", "")),
            source_video=str(row.get("source_video", "")),
            original_path=str(row.get("original_path", "")),
            merged_filename=str(row.get("merged_filename", "")),
            clip_length=int(row.get("clip_length", video.shape[1])),
            height=int(row.get("height", video.shape[-2])),
            width=int(row.get("width", video.shape[-1])),
            dtype=str(row.get("dtype", "uint8")),
            preprocess_type=str(row.get("preprocess_type", "")),
            frame_count=int(row.get("frame_count", 0)),
            sample_fps=float(row.get("sample_fps", 0.0)),
            sha256=str(row.get("sha256", "")),
        )
        return video, label, metadata

    def get_class_distribution(self) -> Dict[int, int]:
        return {int(k): int(v) for k, v in self.manifest["label"].value_counts().to_dict().items()}

    def get_split_info(self) -> Dict[str, Any]:
        return {
            "manifest_path": str(self.manifest_path),
            "total_clips": len(self),
            "split": self.split,
            "class_distribution": self.get_class_distribution(),
            "source_datasets": self.manifest["source_dataset"].value_counts().to_dict()
            if "source_dataset" in self.manifest.columns
            else {},
        }


def collate_fn_clips(batch):
    videos = torch.stack([item[0] for item in batch], dim=0)
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    metadata = [item[2] for item in batch]
    return videos, labels, metadata
