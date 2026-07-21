"""
Model-specific adapters for JRTIP GPU clip caches.

These adapters let existing baselines consume tensors from result/gpu_cache
instead of decoding video or reusing the older Ver1 cache path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from data.processors.gpu_clip_cache_dataset import GpuClipCacheDataset, assert_new_cache_path


def _profile_exists(cache_root: str | Path, profile: str) -> bool:
    return (Path(cache_root) / profile / "manifest.csv").exists()


def _model_specific_profile(
    model_key: str,
    clip_length: int,
    cache_root: str | Path,
) -> tuple[str | None, int, Optional[int]]:
    """Return a model-specific cache profile when it exists."""
    clip_length = int(clip_length)
    candidates: dict[str, tuple[str, int, Optional[int]]] = {
        "c3d": (f"c3d_rgb_t{clip_length}_112", 112, None),
        "slowfast": (f"slowfast_rgb_t{clip_length}_224", 224, None),
        "josenet": (f"josenet_rgb_t{clip_length}_224", 224, None),
    }
    profile = candidates.get(model_key)
    if profile is None:
        return None, 224, None
    name, size, output_size = profile
    if _profile_exists(cache_root, name):
        return name, size, output_size
    return None, 224, output_size


def _resize_clip(video: torch.Tensor, output_size: Optional[int]) -> torch.Tensor:
    """Resize (C,T,H,W) clip to square output_size using trilinear interpolation."""
    if output_size is None or int(video.shape[-1]) == int(output_size):
        return video
    x = video.unsqueeze(0)  # (1,C,T,H,W)
    x = F.interpolate(
        x,
        size=(int(video.shape[1]), int(output_size), int(output_size)),
        mode="trilinear",
        align_corners=False,
    )
    return x.squeeze(0)


def _uniform_temporal_sample(video: torch.Tensor, frames: int) -> torch.Tensor:
    """Uniformly sample/pad a (C,T,H,W) clip to the requested frame count."""
    current = int(video.shape[1])
    frames = int(frames)
    if current == frames:
        return video
    if current > frames:
        idx = torch.linspace(0, current - 1, frames).round().long()
        return video[:, idx, :, :]
    pad = video[:, -1:, :, :].repeat(1, frames - current, 1, 1)
    return torch.cat([video, pad], dim=1)


class CachedVideoDataset(Dataset):
    """
    Generic cached video dataset returning (video, label).

    The underlying cache stores tensors as uint8 (C,T,H,W). This wrapper returns
    float32 tensors, optionally ImageNet-normalized via GpuClipCacheDataset.
    """

    def __init__(
        self,
        cache_root: str | Path = "result/gpu_cache",
        profile: Optional[str] = None,
        clip_length: int = 16,
        split: str = "train",
        preprocess_type: str = "wholeframe",
        size: int = 224,
        output_size: Optional[int] = None,
        normalize: bool = True,
        device: Optional[str | torch.device] = None,
        manifest_path: Optional[str | Path] = None,
    ) -> None:
        assert_new_cache_path(cache_root, "cache_root")
        self.base = GpuClipCacheDataset(
            manifest_path=manifest_path,
            cache_root=cache_root,
            profile=profile,
            clip_length=clip_length,
            preprocess_type=preprocess_type,
            size=size,
            split=split,
            normalize=normalize,
            device=device,
        )
        self.output_size = output_size

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        video, label, metadata = self.base[idx]
        video = _resize_clip(video, self.output_size)
        return video.contiguous(), int(label), {
            "video_id": metadata.video_id,
            "source_video": metadata.source_video,
            "source_dataset": metadata.source_dataset,
            "split": metadata.split,
        }


class SlowFastCacheDataset(Dataset):
    """
    Cached SlowFast dataset returning ((slow, fast), label).

    `fast` is sampled to fast_frames. `slow` is sampled uniformly from `fast`.
    Both tensors have shape (C,T,H,W).
    """

    def __init__(
        self,
        cache_root: str | Path = "result/gpu_cache",
        profile: Optional[str] = None,
        clip_length: int = 32,
        split: str = "train",
        preprocess_type: str = "wholeframe",
        size: int = 224,
        slow_frames: int = 8,
        fast_frames: int = 32,
        normalize: bool = True,
        device: Optional[str | torch.device] = None,
        manifest_path: Optional[str | Path] = None,
    ) -> None:
        assert_new_cache_path(cache_root, "cache_root")
        self.base = GpuClipCacheDataset(
            manifest_path=manifest_path,
            cache_root=cache_root,
            profile=profile,
            clip_length=clip_length,
            preprocess_type=preprocess_type,
            size=size,
            split=split,
            normalize=normalize,
            device=device,
        )
        self.slow_frames = int(slow_frames)
        self.fast_frames = int(fast_frames)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        video, label, metadata = self.base[idx]
        fast = _uniform_temporal_sample(video, self.fast_frames)
        slow = _uniform_temporal_sample(fast, self.slow_frames)
        return (slow.contiguous(), fast.contiguous()), int(label), {
            "video_id": metadata.video_id,
            "source_video": metadata.source_video,
            "source_dataset": metadata.source_dataset,
            "split": metadata.split,
        }


def _standardize_clip(video: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = video.mean()
    std = video.std()
    if float(std) < eps:
        return video - mean
    return (video - mean) / std


def _rgb_u8_from_video(video: torch.Tensor) -> np.ndarray:
    rgb = video.detach().cpu().clamp(0, 1).permute(1, 2, 3, 0).numpy()
    return (rgb * 255.0).round().astype(np.uint8)


def _farneback_flow_from_rgb_u8(rgb_u8: np.ndarray) -> np.ndarray:
    """
    Generate raw Farneback optical flow from cached RGB frames.

    The official JOSENet preprocessing computes Farneback flow between adjacent
    frames and appends a zero flow frame. This keeps the new GPU cache as the
    only video source while avoiding legacy decoded-frame folders.
    """
    try:
        import cv2
    except Exception as exc:  # pragma: no cover - depends on runtime env
        raise RuntimeError("JOSENet requires opencv-python/cv2 to generate optical flow from cache.") from exc

    flows: list[np.ndarray] = []
    for idx in range(max(0, rgb_u8.shape[0] - 1)):
        prev_gray = cv2.cvtColor(rgb_u8[idx], cv2.COLOR_RGB2GRAY)
        next_gray = cv2.cvtColor(rgb_u8[idx + 1], cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flows.append(flow.astype(np.float32))
    flows.append(np.zeros((*rgb_u8.shape[1:3], 2), dtype=np.float32))
    return np.stack(flows, axis=0).astype(np.float32)


def _josenet_roi_video(rgb_u8: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Apply the official JOSENet motion ROI crop/resize to an RGB segment."""
    try:
        import cv2
    except Exception as exc:  # pragma: no cover - depends on runtime env
        raise RuntimeError("JOSENet ROI preprocessing requires opencv-python/cv2.") from exc

    np.random.seed(8)
    magnitudes: list[np.ndarray] = []
    for item in flow:
        f = item.copy()
        f[..., 0] = cv2.normalize(f[..., 0], None, 0, 255, cv2.NORM_MINMAX)
        f[..., 1] = cv2.normalize(f[..., 1], None, 0, 255, cv2.NORM_MINMAX)
        f[:, :, 0] -= np.mean(f[:, :, 0])
        f[:, :, 1] -= np.mean(f[:, :, 1])
        magnitudes.append(np.sqrt(f[:, :, 0] ** 2 + f[:, :, 1] ** 2))

    magnitude = np.sum(magnitudes, axis=0)
    threshold = np.mean(magnitude)
    magnitude[magnitude < threshold] = 0
    x_pdf = np.sum(magnitude, axis=1) + 0.001
    y_pdf = np.sum(magnitude, axis=0) + 0.001
    x_pdf /= np.sum(x_pdf)
    y_pdf /= np.sum(y_pdf)
    x_points = np.random.choice(a=np.arange(224), size=10, replace=True, p=x_pdf)
    y_points = np.random.choice(a=np.arange(224), size=10, replace=True, p=y_pdf)
    x = max(56, min(int(np.mean(x_points)), 167))
    y = max(56, min(int(np.mean(y_points)), 167))

    roi = rgb_u8[:, x - 56 : x + 56, y - 56 : y + 56, :]
    return np.stack([cv2.resize(frame, (224, 224), interpolation=cv2.INTER_CUBIC) for frame in roi], axis=0)


class JOSENetCacheDataset(Dataset):
    """
    Cached JOSENet dataset returning ((rgb, flow), label).

    JOSENet's public architecture is fixed to 16-frame inputs. The source
    remains the new cache only.
    """

    effective_frames = 16

    def __init__(
        self,
        cache_root: str | Path = "result/gpu_cache",
        profile: Optional[str] = None,
        clip_length: int = 16,
        split: str = "train",
        preprocess_type: str = "wholeframe",
        size: int = 224,
        device: Optional[str | torch.device] = None,
        manifest_path: Optional[str | Path] = None,
    ) -> None:
        assert_new_cache_path(cache_root, "cache_root")
        if int(clip_length) != self.effective_frames:
            raise ValueError("JOSENet official architecture supports only 16-frame clips.")
        self.base = GpuClipCacheDataset(
            manifest_path=manifest_path,
            cache_root=cache_root,
            profile=profile,
            clip_length=clip_length,
            preprocess_type=preprocess_type,
            size=size,
            split=split,
            normalize=False,
            device=device,
        )

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        video, label, metadata = self.base[idx]
        video = _uniform_temporal_sample(video, self.effective_frames)
        rgb_u8 = _rgb_u8_from_video(video)
        flow_np = _farneback_flow_from_rgb_u8(rgb_u8)
        rgb_roi = _josenet_roi_video(rgb_u8, flow_np)
        rgb = _standardize_clip(
            torch.from_numpy(rgb_roi).permute(3, 0, 1, 2).float().contiguous()
        )
        flow = _standardize_clip(
            torch.from_numpy(flow_np).permute(3, 0, 1, 2).float().contiguous()
        )
        return (rgb.contiguous(), flow.contiguous()), int(label), {
            "video_id": metadata.video_id,
            "source_video": metadata.source_video,
            "source_dataset": metadata.source_dataset,
            "split": metadata.split,
        }


def _stack_inputs(values: list[Any]) -> Any:
    """Stack tensor or nested tuple inputs while preserving model input structure."""
    first = values[0]
    if isinstance(first, torch.Tensor):
        return torch.stack(values, dim=0)
    if isinstance(first, (tuple, list)):
        return tuple(_stack_inputs([value[index] for value in values]) for index in range(len(first)))
    raise TypeError(f"Unsupported cached model input type: {type(first)!r}")


def _collate_cached_batch(batch: list[Any]) -> Any:
    """Collate cached samples and keep per-video metadata aligned with predictions."""
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    if len(batch[0]) == 2:
        inputs, labels = zip(*batch)
        return _stack_inputs(list(inputs)), torch.tensor(labels, dtype=torch.long)
    inputs, labels, metadata = zip(*batch)
    return (
        _stack_inputs(list(inputs)),
        torch.tensor(labels, dtype=torch.long),
        list(metadata),
    )


def build_cached_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """Build a DataLoader for cached datasets with conservative defaults."""
    kwargs = {
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "drop_last": bool(drop_last),
        "collate_fn": _collate_cached_batch,
    }
    if int(num_workers) > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def make_model_cache_dataset(
    model_name: str,
    split: str,
    cache_root: str | Path = "result/gpu_cache",
    clip_length: int = 16,
    normalize: bool = True,
    manifest_path: Optional[str | Path] = None,
):
    """
    Factory for the current model set.

    Supported model_name values:
    c3d, i3d, resnet_lstm, movinet, videomamba, swin3d, josenet, slowfast.
    """
    assert_new_cache_path(cache_root, "cache_root")
    key = model_name.lower().replace("-", "_")
    profile, cache_size, model_output_size = _model_specific_profile(key, int(clip_length), cache_root)
    if key == "slowfast":
        if int(clip_length) < 32:
            raise ValueError("SlowFast cache adapter requires clip_length 32 or 64.")
        return SlowFastCacheDataset(
            cache_root=cache_root,
            manifest_path=manifest_path,
            profile=profile,
            clip_length=int(clip_length),
            split=split,
            size=cache_size,
            normalize=normalize,
        )
    if key == "josenet":
        return JOSENetCacheDataset(
            cache_root=cache_root,
            manifest_path=manifest_path,
            profile=profile,
            clip_length=int(clip_length),
            split=split,
            size=cache_size,
        )
    if key == "c3d" and int(clip_length) != 16:
        raise ValueError("C3D cache adapter supports only 16-frame clips.")

    output_size = model_output_size if model_output_size is not None else (112 if key == "c3d" else 224)
    if key not in {
        "c3d",
        "i3d",
        "resnet_lstm",
        "movinet",
        "videomamba",
        "swin3d",
        "josenet",
    }:
        raise ValueError(f"Unsupported model_name: {model_name}")

    return CachedVideoDataset(
        cache_root=cache_root,
        manifest_path=manifest_path,
        profile=profile,
        clip_length=int(clip_length),
        split=split,
        size=cache_size,
        output_size=output_size,
        normalize=normalize,
    )
