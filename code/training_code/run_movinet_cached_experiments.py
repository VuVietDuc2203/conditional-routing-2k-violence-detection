"""
Run MoViNet cached experiments for JRTIP (M1/M2/M3 variants).

This script trains/evaluates MoViNet from result/gpu_cache.

Usage:
  python -m training_code.run_movinet_cached_experiments --variant M3 --clip-length 50 --cache-root result/gpu_cache --epochs 30 --batch-size 16
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Path setup
REPO_ROOT = Path(__file__).resolve().parents[1]
# Add project root for data.* imports
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# Add MoViNet path
MOVINET_PATH = REPO_ROOT / "training_code" / "train_movinet_violence" / "MoViNet-pytorch"
if str(MOVINET_PATH) not in sys.path:
    sys.path.insert(0, str(MOVINET_PATH))
MOVINET_A2_PRETRAINED_V2 = MOVINET_PATH / "weights" / "modelA2_statedict_v2"

log = logging.getLogger("run_movinet_cached")

try:
    from movinets import MoViNet
    from movinets.config import _C as movinet_cfg
except ImportError as e:
    log.error("Failed to import MoViNet: %s", e)
    log.error("Ensure MoViNet-pytorch submodule is initialized at %s", MOVINET_PATH)
    sys.exit(1)

# Import cache dataset
try:
    from data.processors.gpu_clip_cache_dataset import GpuClipCacheDataset
except ImportError as e:
    log.error("Failed to import GPU clip cache dataset from data.processors: %s", e)
    log.error("Ensure data/processors/ is a Python package with __init__.py")
    sys.exit(1)

# Check sklearn availability (required for metrics)
try:
    from sklearn.metrics import balanced_accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    precision_recall_fscore_support = None
    confusion_matrix = None

# ==== Constants ====
NUM_CLASSES = 2
LABEL_TO_INT = {"non_violence": 0, "violence": 1}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def collate_fn(batch):
    """Custom collate for GpuClipCacheDataset returning (video, label, metadata)."""
    videos = torch.stack([item[0] for item in batch], dim=0)
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    metadata = [item[2] for item in batch]
    return videos, labels, metadata

# Model variant configurations
VARIANT_CONFIGS = {
    "M1": {
        "name": "MoViNet Baseline (current code)",
        "use_amp": False,
        "use_class_weights": True,
        "use_gradient_clip": True,
        "freeze_bn": True,
        "freeze_backbone": False,
        "optimizer": "adam",
        "lr": 5e-5,
        "weight_decay": 0.0,
    },
    "M2": {
        "name": "MoViNet Improved Training",
        "use_amp": True,
        "use_class_weights": True,
        "use_gradient_clip": True,
        "freeze_bn": False,
        "freeze_backbone": False,
        "optimizer": "adamw",
        "lr": 5e-5,
        "weight_decay": 1e-4,
    },
    "M3": {
        "name": "MoViNet Full Pipeline (preprocessed)",
        "use_amp": True,
        "use_class_weights": True,
        "use_gradient_clip": True,
        "freeze_bn": False,
        "freeze_backbone": False,
        "optimizer": "adamw",
        "lr": 5e-5,
        "weight_decay": 1e-4,
    },
}

GRADIENT_CLIP_VAL = 1.0
EARLY_STOPPING_PATIENCE = 5
MIN_DELTA = 0.001


def validation_selection_value(metrics: Dict[str, Any], metric: str) -> float:
    """Return the predeclared validation-only checkpoint-selection score."""
    if metric == "balanced_composite":
        return 0.5 * float(metrics["balanced_accuracy"]) + 0.5 * float(metrics["f1_macro"])
    return float(metrics[metric])


# ==== Dataset Factory ====
def make_dataset(
    cache_root: Path,
    variant: str,
    clip_length: int,
    split: str,
    size: int = 224,
    profile: str | None = None,
    manifest_path: Path | None = None,
    transform: Any | None = None,
) -> GpuClipCacheDataset:
    """
    Create dataset for the given variant and split.

    M1/M2: preprocess_type='wholeframe'
    M3: preprocess_type='movinet_preprocessed'
    """
    if variant in ("M1", "M2"):
        preprocess_type = "wholeframe"
    elif variant == "M3":
        preprocess_type = "movinet_preprocessed"
    else:
        raise ValueError(f"Unknown variant: {variant}")

    dataset = GpuClipCacheDataset(
        manifest_path=manifest_path,
        cache_root=cache_root,
        profile=profile,
        clip_length=clip_length,
        preprocess_type=preprocess_type,
        size=size,
        split=split,
        normalize=False,
        transform=transform,
    )

    log.info("Created cached dataset for variant=%s split=%s preprocess=%s samples=%d",
             variant, split, preprocess_type, len(dataset))
    return dataset


class TemporalConsistentAugment:
    """Apply one spatial/color transform consistently to every frame in a clip."""

    def __init__(
        self,
        probability: float = 0.0,
        horizontal_flip: bool = True,
        crop_scale_min: float = 1.0,
        brightness: float = 0.0,
        contrast: float = 0.0,
    ) -> None:
        if not 0.0 <= probability <= 1.0:
            raise ValueError("augmentation probability must be in [0, 1]")
        if not 0.5 <= crop_scale_min <= 1.0:
            raise ValueError("crop_scale_min must be in [0.5, 1.0]")
        self.probability = float(probability)
        self.horizontal_flip = bool(horizontal_flip)
        self.crop_scale_min = float(crop_scale_min)
        self.brightness = float(brightness)
        self.contrast = float(contrast)

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        if self.probability <= 0.0 or torch.rand(()) >= self.probability:
            return video
        output = video
        if self.horizontal_flip and torch.rand(()) < 0.5:
            output = torch.flip(output, dims=[-1])

        if self.crop_scale_min < 1.0:
            height, width = int(output.shape[-2]), int(output.shape[-1])
            scale = float(torch.empty(()).uniform_(self.crop_scale_min, 1.0))
            crop_h = max(1, min(height, int(round(height * scale))))
            crop_w = max(1, min(width, int(round(width * scale))))
            top = int(torch.randint(0, height - crop_h + 1, ()).item())
            left = int(torch.randint(0, width - crop_w + 1, ()).item())
            cropped = output[..., top:top + crop_h, left:left + crop_w]
            # MoViNet clips are [C,T,H,W]; treat T as the interpolation batch.
            cropped = cropped.permute(1, 0, 2, 3)
            cropped = F.interpolate(cropped, size=(height, width), mode="bilinear", align_corners=False)
            output = cropped.permute(1, 0, 2, 3)

        if self.contrast > 0.0:
            factor = 1.0 + float(torch.empty(()).uniform_(-self.contrast, self.contrast))
            output = output * factor
        if self.brightness > 0.0:
            offset = float(torch.empty(()).uniform_(-self.brightness, self.brightness))
            output = output + offset
        return output.clamp_(0.0, 1.0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int, deterministic: bool) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def measure_cache_read_throughput(dataset: Dataset, max_samples: int = 16) -> dict[str, float | int]:
    samples = min(int(max_samples), len(dataset))
    if samples <= 0:
        return {"samples": 0, "clips_per_sec": 0.0, "mean_read_ms_per_clip": 0.0, "wall_time_sec": 0.0}
    start = time.perf_counter()
    for idx in range(samples):
        _ = dataset[idx]
    wall_time = time.perf_counter() - start
    return {
        "samples": int(samples),
        "clips_per_sec": float(samples / max(wall_time, 1e-9)),
        "mean_read_ms_per_clip": float((wall_time / samples) * 1000.0),
        "wall_time_sec": float(wall_time),
    }


def require_existing_path(path: str | Path, field_name: str) -> Path:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"{field_name} not found: {path}")
    return resolved


def require_cache_root(path: str | Path) -> Path:
    resolved = require_existing_path(path, "cache_root")
    parts = [part.lower() for part in resolved.parts]
    ok = any(parts[i] == "result" and i + 1 < len(parts) and parts[i + 1] == "gpu_cache" for i in range(len(parts)))
    if not ok and resolved.as_posix().replace("\\", "/") != "result/gpu_cache":
        raise ValueError(f"cache_root must point to result/gpu_cache: {path}")
    return resolved


# ==== Model ====
def load_torch_state_dict(path: Path) -> Dict[str, torch.Tensor]:
    """Load a plain PyTorch state_dict from a local checkpoint path."""
    if not path.exists():
        raise FileNotFoundError(f"MoViNet pretrained weights not found: {path}")
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Expected state_dict checkpoint at {path}, got {type(state).__name__}")
    return state


def adapt_movinet_a2_v2_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Translate legacy MoViNet v2 checkpoint keys to the current module names."""
    adapted: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        parts = key.split(".")
        new_key = key

        if parts[0] in {"conv1", "conv7"} and len(parts) >= 3 and parts[1] in {"conv3d", "norm"}:
            new_key = ".".join([parts[0], "conv_1", *parts[1:]])
        elif parts[0] == "classifier" and len(parts) == 3 and parts[1] in {"0", "3"}:
            new_key = ".".join([parts[0], parts[1], "conv_1", "conv3d", parts[2]])
        elif parts[0] == "blocks" and len(parts) >= 5 and "se" in parts:
            if parts[-2] in {"fc1", "fc2"} and parts[-1] in {"weight", "bias"}:
                new_key = ".".join([*parts[:-1], "conv_1", "conv3d", parts[-1]])
        elif parts[0] == "blocks":
            for idx, part in enumerate(parts):
                if part in {"conv3d", "norm"}:
                    new_key = ".".join([*parts[:idx], "conv_1", *parts[idx:]])
                    break

        adapted[new_key] = value
    return adapted


def create_model(device: torch.device) -> torch.nn.Module:
    """Create MoViNetA2 model with 2-class classifier."""
    model = MoViNet(
        movinet_cfg.MODEL.MoViNetA2,
        causal=False,
        pretrained=False,
        num_classes=600,
        conv_type="3d",
        tf_like=True,
    )
    state_dict = load_torch_state_dict(MOVINET_A2_PRETRAINED_V2)
    state_dict = adapt_movinet_a2_v2_state_dict(state_dict)
    model.load_state_dict(state_dict, strict=True)
    log.info("Loaded MoViNetA2 pretrained weights from %s", MOVINET_A2_PRETRAINED_V2)
    feature_dim = movinet_cfg.MODEL.MoViNetA2.dense9.hidden_dim
    model.classifier[3] = torch.nn.Conv3d(feature_dim, NUM_CLASSES, (1, 1, 1))
    model = model.to(device)
    return model


def freeze_movinet_backbone(model: torch.nn.Module) -> Tuple[int, int]:
    """Freeze all MoViNet parameters except the replaced task classifier."""
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
    model.freeze_backbone = True
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return trainable, total


def keep_movinet_backbone_eval(model: torch.nn.Module) -> None:
    """Keep frozen MoViNet blocks in eval mode while the classifier head trains."""
    if not bool(getattr(model, "freeze_backbone", False)):
        return
    model.eval()
    model.classifier.train()


def freeze_batchnorm_layers(model: torch.nn.Module) -> None:
    """Freeze BatchNorm affine parameters and running statistics."""
    for module in model.modules():
        if isinstance(module, (torch.nn.BatchNorm3d, torch.nn.BatchNorm2d)):
            module.eval()
            module.weight.requires_grad = False
            module.bias.requires_grad = False


def keep_batchnorm_eval(model: torch.nn.Module) -> None:
    """Keep frozen BatchNorm statistics fixed after model.train() is called."""
    for module in model.modules():
        if isinstance(module, (torch.nn.BatchNorm3d, torch.nn.BatchNorm2d)):
            module.eval()


# ==== Training / Eval ====
def train_epoch(
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    use_gradient_clip: bool,
    class_weights: Optional[torch.Tensor],
    scaler: Optional[torch.amp.GradScaler],
    freeze_bn: bool,
    label_smoothing: float = 0.0,
) -> float:
    """Run one training epoch. Returns average loss."""
    model.train()
    if freeze_bn:
        keep_batchnorm_eval(model)
    keep_movinet_backbone_eval(model)
    model.clean_activation_buffers()
    optimizer.zero_grad()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (videos, labels, _metadata) in enumerate(loader):
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if use_amp and scaler is not None:
            with torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda"):
                outputs = model(videos)
                loss = F.cross_entropy(
                    outputs, labels, weight=class_weights, label_smoothing=label_smoothing
                )
            scaler.scale(loss).backward()
            if use_gradient_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VAL)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(videos)
            loss = F.cross_entropy(outputs, labels, weight=class_weights)
            loss.backward()
            if use_gradient_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VAL)
            optimizer.step()

        optimizer.zero_grad()
        model.clean_activation_buffers()
        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    return_details: bool = True,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Evaluate model and return metrics."""
    model.eval()
    model.clean_activation_buffers()

    all_preds: List[int] = []
    all_labels: List[int] = []
    all_scores: List[float] = []
    all_metadata: List[Dict[str, Any]] = []
    total_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        for videos, labels, metadata in loader:
            videos = videos.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if use_amp:
                with torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda"):
                    outputs = model(videos)
                    loss = F.cross_entropy(outputs, labels, reduction="sum")
            else:
                outputs = model(videos)
                loss = F.cross_entropy(outputs, labels, reduction="sum")

            total_loss += loss.item()
            num_samples += labels.size(0)

            probs = torch.softmax(outputs, dim=1)
            preds = (probs[:, 1] >= float(threshold)).to(torch.long)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_scores.extend(probs[:, 1].detach().cpu().numpy().tolist())
            for m in metadata:
                all_metadata.append({
                    "video_id": getattr(m, "video_id", ""),
                    "source_video": getattr(m, "source_video", ""),
                    "split": getattr(m, "split", ""),
                })

            model.clean_activation_buffers()

    avg_loss = total_loss / max(num_samples, 1)
    all_labels_np = np.array(all_labels)
    all_preds_np = np.array(all_preds)

    accuracy = (all_preds_np == all_labels_np).mean()

    # Compute precision, recall, F1 (macro)
    from sklearn.metrics import balanced_accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels_np, all_preds_np, average="macro", zero_division=0
    )
    f1_binary = f1_score(all_labels_np, all_preds_np, zero_division=0)
    balanced_accuracy = balanced_accuracy_score(all_labels_np, all_preds_np)
    cm = confusion_matrix(all_labels_np, all_preds_np, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

    result = {
        "loss": avg_loss,
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1_binary),
        "f1_macro": float(f1),
        "num_samples": int(num_samples),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "classification_threshold": float(threshold),
    }

    if return_details:
        result["predictions"] = [
            {
                "video_id": m["video_id"],
                "source_video": m["source_video"],
                "true": int(l),
                "pred": int(p),
                "score_violence": float(s),
            }
            for m, l, p, s in zip(all_metadata, all_labels, all_preds, all_scores)
        ]

    return result


def measure_peak_vram() -> float:
    """Return peak VRAM usage in MB."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def reset_peak_vram():
    """Reset peak VRAM measurement."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def require_cuda_device(device_arg: str) -> torch.device:
    if not str(device_arg).startswith("cuda"):
        raise RuntimeError("MoViNet cached training is GPU-only for this plan. Use --device cuda.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; refusing to train MoViNet on CPU.")
    return torch.device(device_arg)


def parse_batch_size(raw: str | int) -> int | str:
    if isinstance(raw, int):
        return int(raw)
    value = str(raw).strip().lower()
    if value == "auto":
        return "auto"
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("--batch-size must be a positive integer or 'auto'.")
    return parsed


def is_oom_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in text or "cuda error: out of memory" in text


def max_vram_mb(device: torch.device, target_ratio: float) -> float:
    props = torch.cuda.get_device_properties(device)
    return float(props.total_memory / (1024 * 1024) * float(target_ratio))


def try_movinet_batch_size(
    model: torch.nn.Module,
    dataset: Dataset,
    batch_size: int,
    device: torch.device,
    use_amp: bool,
    class_weights: Optional[torch.Tensor],
    freeze_bn: bool,
    vram_target_mb: float,
) -> tuple[bool, float, str]:
    reset_peak_vram()
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn,
    )
    try:
        videos, labels, _metadata = next(iter(loader))
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        model.train()
        if freeze_bn:
            keep_batchnorm_eval(model)
        keep_movinet_backbone_eval(model)
        model.clean_activation_buffers()
        model.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=use_amp and device.type == "cuda"):
            outputs = model(videos)
            loss = F.cross_entropy(
                outputs, labels, weight=class_weights, label_smoothing=label_smoothing
            )
        loss.backward()
        peak = measure_peak_vram()
        model.clean_activation_buffers()
        model.zero_grad(set_to_none=True)
        ok = peak <= vram_target_mb
        reason = "" if ok else f"peak_vram_mb={peak:.0f} exceeds target={vram_target_mb:.0f}"
        del videos, labels, outputs, loss
        torch.cuda.empty_cache()
        return ok, peak, reason
    except Exception as exc:
        model.clean_activation_buffers()
        model.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if is_oom_error(exc):
            return False, measure_peak_vram(), "cuda_oom"
        raise


def auto_tune_batch_size(
    model: torch.nn.Module,
    dataset: Dataset,
    device: torch.device,
    use_amp: bool,
    class_weights: Optional[torch.Tensor],
    freeze_bn: bool,
    vram_target: float,
    max_auto_batch: int,
) -> int:
    target_mb = max_vram_mb(device, vram_target)
    max_auto_batch = max(1, int(max_auto_batch))
    good = 0
    bad = max_auto_batch + 1
    candidate = 1
    last_peak = 0.0
    while candidate <= max_auto_batch:
        ok, peak, reason = try_movinet_batch_size(
            model, dataset, candidate, device, use_amp, class_weights, freeze_bn, target_mb
        )
        last_peak = peak
        log.info(
            "auto_batch probe batch=%d ok=%s peak_vram=%.0fMB target=%.0fMB%s",
            candidate,
            ok,
            peak,
            target_mb,
            f" reason={reason}" if reason else "",
        )
        if ok:
            good = candidate
            candidate *= 2
        else:
            bad = candidate
            break
    if good <= 0:
        raise RuntimeError("Auto-batch failed even at batch_size=1.")
    low = good + 1
    high = min(bad - 1, max_auto_batch)
    while low <= high:
        mid = (low + high) // 2
        ok, peak, reason = try_movinet_batch_size(
            model, dataset, mid, device, use_amp, class_weights, freeze_bn, target_mb
        )
        last_peak = peak
        log.info(
            "auto_batch binary batch=%d ok=%s peak_vram=%.0fMB target=%.0fMB%s",
            mid,
            ok,
            peak,
            target_mb,
            f" reason={reason}" if reason else "",
        )
        if ok:
            good = mid
            low = mid + 1
        else:
            high = mid - 1
    log.info("auto_batch selected batch_size=%d target_ratio=%.2f last_peak=%.0fMB", good, vram_target, last_peak)
    return int(good)


def collect_dataset_labels(dataset: Dataset) -> List[int]:
    """Collect labels without loading video tensors when possible."""
    if isinstance(dataset, torch.utils.data.Subset):
        base = dataset.dataset
        indices = dataset.indices
        if hasattr(base, "manifest"):
            return [int(base.manifest.iloc[int(i)]["label"]) for i in indices]
    if hasattr(dataset, "manifest"):
        return [int(x) for x in dataset.manifest["label"].tolist()]
    labels = []
    for i in range(len(dataset)):
        item = dataset[i]
        labels.append(int(item[1]))
    return labels


# ==== Main ====
def main():
    parser = argparse.ArgumentParser(
        description="Run MoViNet cached experiments (M1/M2/M3) for JRTIP"
    )
    parser.add_argument(
        "--variant",
        choices=["M1", "M2", "M3"],
        required=True,
        help="Model variant: M1=baseline, M2=improved train, M3=full pipeline",
    )
    parser.add_argument(
        "--clip-length",
        type=int,
        choices=[16, 32, 50, 64],
        required=True,
        help="Number of frames per clip",
    )
    parser.add_argument("--cache-root", type=Path, default=Path("result/gpu_cache"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Frozen manifest override containing train/val/test rows for this variant.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("result/movinet_cached_experiments"),
        help="Output directory for results",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", default="16", help="Positive integer or 'auto'.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default=None)
    parser.add_argument(
        "--freeze-bn",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the variant BatchNorm freeze policy.",
    )
    parser.add_argument(
        "--class-weights",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the variant class-weight policy.",
    )
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument(
        "--selection-metric",
        choices=["accuracy", "balanced_accuracy", "f1", "f1_macro", "balanced_composite"],
        default="f1",
    )
    parser.add_argument("--classification-threshold", type=float, default=0.5)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-patience", type=int, default=2)
    parser.add_argument("--augment-prob", type=float, default=0.0)
    parser.add_argument("--crop-scale-min", type=float, default=1.0)
    parser.add_argument("--brightness", type=float, default=0.0)
    parser.add_argument("--contrast", type=float, default=0.0)
    parser.add_argument(
        "--development-only",
        action="store_true",
        help="Train/evaluate validation only; requires a manifest containing no test rows.",
    )
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)
    parser.add_argument("--limit", type=int, default=None, help="Limit dataset size (smoke test)")
    parser.add_argument("--smoke", action="store_true", help="Run smoke test (1 epoch, small batch)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true", help="Force AMP (overrides variant config)")
    parser.add_argument("--resume", type=Path, default=None, help="Resume from checkpoint")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate, no training")
    parser.add_argument("--throughput-samples", type=int, default=16)
    parser.add_argument("--vram-target", type=float, default=0.92)
    parser.add_argument("--max-auto-batch", type=int, default=128)
    parser.add_argument("--seed", type=int, default=50900)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use deterministic CuDNN behavior (default: enabled).",
    )
    args = parser.parse_args()

    if not 0.0 <= args.label_smoothing < 1.0:
        parser.error("--label-smoothing must be in [0, 1)")
    if not 0.0 < args.classification_threshold < 1.0:
        parser.error("--classification-threshold must be in (0, 1)")
    if not 0.0 < args.scheduler_factor < 1.0:
        parser.error("--scheduler-factor must be in (0, 1)")

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    # Validate cache input
    try:
        args.cache_root = require_cache_root(args.cache_root)
    except (ValueError, FileNotFoundError) as e:
        log.error(e)
        sys.exit(1)
    if args.manifest is not None:
        args.manifest = require_existing_path(args.manifest, "manifest")
        manifest_header = pd.read_csv(args.manifest, nrows=1)
        manifest_columns = set(manifest_header.columns)
        required_columns = {"cache_path", "video_id", "label", "split"}
        missing_columns = sorted(required_columns - manifest_columns)
        if missing_columns:
            log.error("Manifest is missing required columns: %s", missing_columns)
            sys.exit(1)
        if args.development_only:
            development_splits = set(
                pd.read_csv(args.manifest, usecols=["split"])["split"].astype(str).unique()
            )
            if development_splits != {"train", "val"}:
                log.error(
                    "Development-only manifest must contain exactly train/val rows; found %s",
                    sorted(development_splits),
                )
                sys.exit(1)
    elif args.development_only:
        log.error("--development-only requires an explicit train/val-only --manifest")
        sys.exit(1)

    data_generator = seed_everything(args.seed, args.deterministic)

    # Resolve variant config
    vcfg = VARIANT_CONFIGS[args.variant]
    use_amp = args.amp or vcfg["use_amp"]
    freeze_backbone = vcfg["freeze_backbone"]
    if freeze_backbone and not args.amp:
        use_amp = False
    use_class_weights = (
        vcfg["use_class_weights"] if args.class_weights is None else bool(args.class_weights)
    )
    use_gradient_clip = vcfg["use_gradient_clip"]
    freeze_bn = vcfg["freeze_bn"] if args.freeze_bn is None else bool(args.freeze_bn)
    optimizer_type = args.optimizer or vcfg["optimizer"]
    lr = args.lr if args.lr is not None else vcfg["lr"]
    weight_decay = args.weight_decay if args.weight_decay is not None else vcfg["weight_decay"]

    # Smoke test overrides
    if args.smoke:
        log.info("Smoke test mode: overriding epochs=1, batch-size=2, limit=8")
        args.epochs = 1
        if str(args.batch_size).lower() == "auto":
            args.batch_size = "2"
        else:
            args.batch_size = min(int(args.batch_size), 2)
        args.limit = args.limit or 8
    # Check prerequisites
    if not SKLEARN_AVAILABLE:
        log.error("scikit-learn is required. Install with: pip install scikit-learn")
        sys.exit(1)

    device = require_cuda_device(args.device)

    # Output directory
    output_dir = args.output_root / f"variant_{args.variant}" / f"t{args.clip_length}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save run config
    run_config = {
        "variant": args.variant,
        "variant_name": vcfg["name"],
        "clip_length": args.clip_length,
        "cache_root": str(args.cache_root),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "optimizer": optimizer_type,
        "label_smoothing": float(args.label_smoothing),
        "selection_metric": args.selection_metric,
        "classification_threshold": float(args.classification_threshold),
        "scheduler_factor": float(args.scheduler_factor),
        "scheduler_patience": int(args.scheduler_patience),
        "augment_prob": float(args.augment_prob),
        "crop_scale_min": float(args.crop_scale_min),
        "brightness": float(args.brightness),
        "contrast": float(args.contrast),
        "development_only": bool(args.development_only),
        "use_amp": use_amp,
        "use_class_weights": use_class_weights,
        "use_gradient_clip": use_gradient_clip,
        "freeze_bn": freeze_bn,
        "freeze_backbone": freeze_backbone,
        "device": args.device,
        "patience": args.patience,
        "vram_target": args.vram_target,
        "max_auto_batch": args.max_auto_batch,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "manifest": str(args.manifest) if args.manifest is not None else None,
        "manifest_sha256": sha256_file(args.manifest) if args.manifest is not None else None,
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    log.info("Starting %s (clip_length=%d) -> %s", vcfg["name"], args.clip_length, output_dir)

    # Create datasets
    log.info("Loading datasets from cache: %s", args.cache_root)
    train_transform = TemporalConsistentAugment(
        probability=args.augment_prob,
        horizontal_flip=True,
        crop_scale_min=args.crop_scale_min,
        brightness=args.brightness,
        contrast=args.contrast,
    ) if args.augment_prob > 0.0 else None
    train_dataset = make_dataset(
        args.cache_root,
        args.variant,
        args.clip_length,
        "train",
        size=224,
        manifest_path=args.manifest,
        transform=train_transform,
    )
    val_dataset = make_dataset(
        args.cache_root,
        args.variant,
        args.clip_length,
        "val",
        size=224,
        manifest_path=args.manifest,
    )
    test_dataset = None
    if not args.development_only:
        test_dataset = make_dataset(
            args.cache_root,
            args.variant,
            args.clip_length,
            "test",
            size=224,
            manifest_path=args.manifest,
        )

    data_profile = "movinet_preprocessed" if args.variant == "M3" else "wholeframe"

    # Apply limit if specified
    if args.limit:
        train_dataset = torch.utils.data.Subset(
            train_dataset, list(range(min(args.limit, len(train_dataset))))
        )
        val_dataset = torch.utils.data.Subset(
            val_dataset, list(range(min(args.limit // 2, len(val_dataset))))
        )
        if test_dataset is not None:
            test_dataset = torch.utils.data.Subset(
                test_dataset, list(range(min(args.limit // 2, len(test_dataset))))
            )

    throughput_dataset = val_dataset if args.development_only else test_dataset
    cache_read_throughput = measure_cache_read_throughput(
        throughput_dataset, max_samples=args.throughput_samples
    )

    log.info(
        "Dataset sizes - Train: %d, Val: %d, Test: %s",
        len(train_dataset),
        len(val_dataset),
        "SKIPPED" if test_dataset is None else str(len(test_dataset)),
    )

    # Device
    args.batch_size = parse_batch_size(args.batch_size)
    log.info("Using device: %s", device)

    # Compute class weights if needed (after device known)
    class_weights = None
    if use_class_weights:
        labels = collect_dataset_labels(train_dataset)
        counts = np.bincount(labels, minlength=NUM_CLASSES)
        weights = len(labels) / (NUM_CLASSES * counts.astype(np.float32))
        class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
        log.info("Class weights: %s", weights)

    # Create model
    model = create_model(device)

    if freeze_backbone:
        trainable_params, total_params = freeze_movinet_backbone(model)
        log.info(
            "Frozen MoViNet backbone: trainable_params=%d/%d",
            trainable_params,
            total_params,
        )

    if freeze_bn:
        freeze_batchnorm_layers(model)
        log.info("Frozen BatchNorm layers")

    if args.batch_size == "auto":
        args.batch_size = auto_tune_batch_size(
            model,
            train_dataset,
            device,
            use_amp,
            class_weights,
            freeze_bn,
            args.vram_target,
            args.max_auto_batch,
        )

    # Create dataloaders
    dl_kwargs = {
        "batch_size": int(args.batch_size),
        "num_workers": args.num_workers,
        "pin_memory": True,
        "collate_fn": collate_fn,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        generator=data_generator,
        **dl_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **dl_kwargs,
    )
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            shuffle=False,
            **dl_kwargs,
        )
    run_config["batch_size"] = int(args.batch_size)
    run_config["num_workers"] = int(args.num_workers)
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    # Optimizer
    trainable_parameters = [param for param in model.parameters() if param.requires_grad]
    if optimizer_type == "adamw":
        optimizer = optim.AdamW(trainable_parameters, lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(trainable_parameters, lr=lr, weight_decay=weight_decay)
    log.info("Optimizer: %s, lr=%.6f, weight_decay=%.6f", optimizer_type, lr, weight_decay)

    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(args.scheduler_factor),
        patience=int(args.scheduler_patience),
    )

    # AMP scaler
    scaler = torch.amp.GradScaler("cuda") if use_amp and device.type == "cuda" else None
    if scaler:
        log.info("Using AMP mixed precision")

    # Resume if needed
    start_epoch = 0
    best_selection_value = float("-inf")
    ckpt: Dict[str, Any] = {}
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_selection_value = float(
            ckpt.get(
                "selection_value",
                ckpt.get("val_metrics", {}).get(args.selection_metric, ckpt.get("val_f1", 0.0)),
            )
        )
        log.info(
            "Resumed from epoch %d, best_%s=%.4f",
            start_epoch,
            args.selection_metric,
            best_selection_value,
        )

    # Training loop
    history = defaultdict(list)
    patience_counter = 0
    peak_vram_overall = 0.0

    if not args.eval_only:
        log.info("Starting training for %d epochs", args.epochs)
        for epoch in range(start_epoch, args.epochs):
            epoch_start = time.time()
            reset_peak_vram()

            train_loss = train_epoch(
                model, optimizer, train_loader, device,
                use_amp, use_gradient_clip, class_weights, scaler, freeze_bn,
                label_smoothing=float(args.label_smoothing),
            )
            train_time = time.time() - epoch_start

            val_metrics = evaluate(model, val_loader, device, use_amp, return_details=False)
            val_f1 = val_metrics["f1"]
            selection_value = validation_selection_value(val_metrics, args.selection_metric)

            peak_vram = measure_peak_vram()
            peak_vram_overall = max(peak_vram_overall, peak_vram)

            scheduler.step(selection_value)

            history["epoch"].append(epoch + 1)
            history["train_loss"].append(train_loss)
            history["train_time_sec"].append(train_time)
            history["val_loss"].append(val_metrics["loss"])
            history["val_accuracy"].append(val_metrics["accuracy"])
            history["val_balanced_accuracy"].append(val_metrics["balanced_accuracy"])
            history["val_precision"].append(val_metrics["precision"])
            history["val_recall"].append(val_metrics["recall"])
            history["val_f1"].append(val_f1)
            history["val_f1_macro"].append(val_metrics["f1_macro"])
            history["peak_vram_mb"].append(peak_vram)

            log.info(
                "Epoch %d: train_loss=%.4f, val_f1=%.4f, val_bal_acc=%.4f, val_acc=%.2f%%, VRAM_peak=%.0fMB, time=%.1fs",
                epoch + 1,
                train_loss,
                val_f1,
                val_metrics["balanced_accuracy"],
                val_metrics["accuracy"] * 100.0,
                peak_vram,
                train_time,
            )

            # Save best model by the predeclared validation metric.
            if selection_value > best_selection_value + MIN_DELTA:
                best_selection_value = selection_value
                patience_counter = 0
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_f1": val_f1,
                    "val_metrics": val_metrics,
                    "selection_metric": args.selection_metric,
                    "selection_value": selection_value,
                }, output_dir / "best.pt")
                log.info(
                    "  >> New best model saved (%s=%.4f)",
                    args.selection_metric,
                    selection_value,
                )
            else:
                patience_counter += 1

            # Save history CSV
            df = pd.DataFrame(history)
            df.to_csv(output_dir / "history.csv", index=False)

            # Early stopping
            if patience_counter >= args.patience:
                log.info("Early stopping triggered after %d epochs without improvement", patience_counter)
                break

    # Load best model for final validation and optional test evaluation.
    best_path = output_dir / "best.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        log.info(
            "Loaded best model from %s (%s=%.4f)",
            best_path,
            ckpt.get("selection_metric", args.selection_metric),
            float(ckpt.get("selection_value", ckpt.get("val_f1", 0.0))),
        )
    else:
        log.warning("No best checkpoint found, using current model state")

    log.info("Running final validation evaluation...")
    validation_metrics = evaluate(
        model,
        val_loader,
        device,
        use_amp,
        return_details=True,
        threshold=float(args.classification_threshold),
    )
    validation_predictions_csv = output_dir / "validation_predictions.csv"
    with open(validation_predictions_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "source_video", "true_label", "pred_label", "score_violence"])
        for pred in validation_metrics["predictions"]:
            writer.writerow([
                pred["video_id"],
                pred["source_video"],
                pred["true"],
                pred["pred"],
                pred["score_violence"],
            ])
    del validation_metrics["predictions"]
    validation_metrics["peak_vram_mb"] = float(peak_vram_overall)
    validation_metrics["batch_size"] = int(args.batch_size)
    validation_metrics["best_epoch"] = int(ckpt.get("epoch", -1) + 1) if ckpt else None
    validation_metrics["selection_metric"] = args.selection_metric
    validation_metrics["selection_value"] = (
        float(
            ckpt.get(
                "selection_value",
                validation_selection_value(ckpt.get("val_metrics", validation_metrics), args.selection_metric),
            )
        )
        if ckpt
        else None
    )
    validation_metrics["manifest"] = str(args.manifest) if args.manifest is not None else None
    validation_metrics["manifest_sha256"] = (
        sha256_file(args.manifest) if args.manifest is not None else None
    )
    (output_dir / "validation_metrics.json").write_text(
        json.dumps(validation_metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("Saved validation outputs to %s", output_dir)

    if args.development_only:
        log.info("Development-only experiment complete; test dataset was never constructed.")
        return

    # Final test evaluation
    log.info("Running test evaluation...")
    if test_loader is None or test_dataset is None:
        raise RuntimeError("Test evaluation requested without a test dataset")
    test_metrics = evaluate(
        model,
        test_loader,
        device,
        use_amp,
        return_details=True,
        threshold=float(args.classification_threshold),
    )

    # Save predictions CSV
    predictions_csv = output_dir / "predictions.csv"
    with open(predictions_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "source_video", "true_label", "pred_label", "score_violence"])
        for pred in test_metrics["predictions"]:
            writer.writerow([
                pred["video_id"],
                pred["source_video"],
                pred["true"],
                pred["pred"],
                pred["score_violence"],
            ])
    log.info("Saved predictions to %s", predictions_csv)

    # Remove predictions from metrics JSON (separate file)
    del test_metrics["predictions"]

    # Add dataset info
    test_metrics["cache_read_throughput"] = cache_read_throughput
    test_metrics["peak_vram_mb"] = float(peak_vram_overall)
    test_metrics["batch_size"] = int(args.batch_size)
    test_metrics["best_epoch"] = int(ckpt.get("epoch", -1) + 1) if best_path.exists() else None
    test_metrics["dataset_info"] = {
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "data_source": "result/gpu_cache",
        "data_profile": data_profile,
        "cache_root": str(args.cache_root),
        "manifest": str(args.manifest) if args.manifest is not None else None,
        "manifest_sha256": sha256_file(args.manifest) if args.manifest is not None else None,
        "seed": int(args.seed),
    }

    # Save metrics JSON
    (output_dir / "metrics.json").write_text(
        json.dumps(test_metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("Saved metrics to %s", output_dir / "metrics.json")

    log.info("Experiment complete. Results in %s", output_dir)
    log.info("Final Test Metrics:")
    log.info("  Accuracy: %.2f%%", test_metrics["accuracy"] * 100.0)
    log.info("  Precision: %.4f", test_metrics["precision"])
    log.info("  Recall: %.4f", test_metrics["recall"])
    log.info("  F1: %.4f", test_metrics["f1"])
    log.info("  Confusion: TN=%d, FP=%d, FN=%d, TP=%d",
             test_metrics["tn"], test_metrics["fp"], test_metrics["fn"], test_metrics["tp"])


if __name__ == "__main__":
    if sys.platform == "win32":
        import multiprocessing
        multiprocessing.freeze_support()
    main()
