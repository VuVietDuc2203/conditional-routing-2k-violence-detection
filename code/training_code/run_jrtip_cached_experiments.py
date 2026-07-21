"""JRTIP cached experiment runner.

This runner reads tensors from result/gpu_cache and does not decode source
videos during training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.processors.model_cache_adapters import (
    build_cached_dataloader,
    make_model_cache_dataset,
)
from training_code.official_video_models import (
    OfficialJOSENetClassifier,
    TorchvisionSwin3DClassifier,
)

log = logging.getLogger("jrtip_cached_experiments")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


LEGACY_MARKERS = (
    "Ver1InferenceDataset",
    "extract_ver1",
    "results/ver1_inference",
    "results\\ver1_inference",
    "data/preprocessed",
    "data\\preprocessed",
    "datasets/preprocessed",
    "datasets\\preprocessed",
)

MODEL_TRAIN_DEFAULTS = {
    "swin3d": {"lr": 8e-5, "weight_decay": 1e-4, "label_smoothing": 0.05},
    "josenet": {"lr": 1e-2, "weight_decay": 1e-4, "label_smoothing": 0.0},
}

ALLOWED_MODEL_CLIP_LENGTHS = {
    "c3d": {16},
    "i3d": {16, 32, 64},
    "resnet_lstm": {16, 32, 64},
    "slowfast": {32, 64},
    "swin3d": {16, 32, 64},
    "josenet": {16},
}

UNAVAILABLE_OFFICIAL_MODELS: dict[str, str] = {}


class C3DClassifier(nn.Module):
    """Classic C3D-style network for 16-frame 112x112 clips."""

    def __init__(self, pretrained: bool = False, freeze_backbone: bool = False) -> None:
        super().__init__()
        if pretrained:
            raise RuntimeError(
                "True C3D pretrained weights are not configured in this repo. "
                "Run C3D without --pretrained, or add an explicit C3D checkpoint loader first."
            )
        def conv_block(in_channels: int, out_channels: int, pool: tuple[int, int, int] | int | None = None) -> list[nn.Module]:
            layers: list[nn.Module] = [
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
            ]
            if pool is not None:
                layers.append(nn.MaxPool3d(kernel_size=pool, stride=pool))
            return layers

        self.features = nn.Sequential(
            *conv_block(3, 64, (1, 2, 2)),
            *conv_block(64, 128, 2),
            *conv_block(128, 256),
            *conv_block(256, 256, 2),
            *conv_block(256, 512),
            *conv_block(512, 512, 2),
            nn.AdaptiveAvgPool3d(1),
        )
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(512, 2),
        )
        self.implementation = "batchnorm_c3d_scratch"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class I3DClassifier(nn.Module):
    """PytorchVideo I3D-ResNet50 backbone."""

    def __init__(self, pretrained: bool = False, freeze_backbone: bool = False) -> None:
        super().__init__()
        from pytorchvideo.models.hub import i3d_r50

        self.backbone = i3d_r50(pretrained=pretrained)
        in_features = self.backbone.blocks[-1].proj.in_features
        self.backbone.blocks[-1].proj = nn.Identity()
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 2),
        )
        self.implementation = "pytorchvideo_i3d_r50"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(x))


class ResNetLSTMClassifier(nn.Module):
    def __init__(
        self,
        pretrained: bool = False,
        freeze_backbone: bool = False,
        hidden_size: int = 256,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)
        in_features = resnet.fc.in_features
        resnet.fc = nn.Identity()
        self.frame_encoder = resnet
        if freeze_backbone:
            for param in self.frame_encoder.parameters():
                param.requires_grad = False
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, frames, height, width = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
        features = self.frame_encoder(x).reshape(batch, frames, -1)
        sequence, _ = self.lstm(features)
        return self.classifier(sequence.mean(dim=1))


class SlowFastClassifier(nn.Module):
    def __init__(self, pretrained: bool = False, freeze_backbone: bool = False) -> None:
        super().__init__()
        from pytorchvideo.models.hub import slowfast_r50

        self.backbone = slowfast_r50(pretrained=pretrained)
        in_features = self.backbone.blocks[-1].proj.in_features
        self.backbone.blocks[-1].proj = nn.Identity()
        self.implementation = "pytorchvideo_slowfast_r50"
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 2),
        )

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        slow, fast = inputs
        features = self.backbone([slow, fast])
        return self.classifier(features)


def reject_legacy_path(path: str | Path, field_name: str) -> None:
    normalized = str(path).replace("\\", "/")
    for marker in LEGACY_MARKERS:
        if marker.replace("\\", "/") in normalized:
            raise ValueError(f"{field_name} points to legacy data/cache: {path}")


def require_existing_path(path: str | Path, field_name: str) -> Path:
    reject_legacy_path(path, field_name)
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


def require_result_output(path: str | Path) -> Path:
    output = Path(path)
    parts = [part.lower() for part in output.parts]
    if "result" not in parts and (not parts or parts[0] != "result"):
        raise ValueError(f"output-root must be under result/: {path}")
    return output


def set_runtime_optimizations(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def reset_peak_vram(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def measure_peak_vram(device: torch.device) -> float:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
        return float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
    return 0.0


def require_cuda_device(device_arg: str | None) -> torch.device:
    requested = device_arg or "cuda"
    if not str(requested).startswith("cuda"):
        raise RuntimeError("Direct training is GPU-only for this plan. Use --device cuda.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; refusing to train on CPU.")
    return torch.device(requested)


def parse_batch_size(raw: str | int | None) -> int | str:
    if raw is None:
        return 4
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


def limit_dataset(dataset: Dataset, limit: int | None) -> Dataset:
    if limit is None or int(limit) <= 0 or int(limit) >= len(dataset):
        return dataset
    return Subset(dataset, list(range(int(limit))))


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


def collect_dataset_labels(dataset: Dataset) -> list[int]:
    """Collect labels without decoding video tensors when possible."""
    if isinstance(dataset, Subset):
        base = dataset.dataset
        if hasattr(base, "base") and hasattr(base.base, "manifest"):
            return [int(base.base.manifest.iloc[int(i)]["label"]) for i in dataset.indices]
        if hasattr(base, "manifest"):
            return [int(base.manifest.iloc[int(i)]["label"]) for i in dataset.indices]
    if hasattr(dataset, "base") and hasattr(dataset.base, "manifest"):
        return [int(x) for x in dataset.base.manifest["label"].tolist()]
    if hasattr(dataset, "manifest"):
        return [int(x) for x in dataset.manifest["label"].tolist()]
    return [int(dataset[i][1]) for i in range(len(dataset))]


def make_class_weights(dataset: Dataset, device: torch.device) -> torch.Tensor:
    labels = collect_dataset_labels(dataset)
    counts = np.bincount(labels, minlength=2).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = len(labels) / (2.0 * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_model(model_name: str, pretrained: bool, freeze_backbone: bool, clip_length: int = 16) -> nn.Module:
    key = model_name.lower().replace("-", "_")
    if key == "c3d":
        return C3DClassifier(pretrained=pretrained, freeze_backbone=freeze_backbone)
    if key == "i3d":
        return I3DClassifier(pretrained=pretrained, freeze_backbone=freeze_backbone)
    if key == "resnet_lstm":
        return ResNetLSTMClassifier(pretrained=pretrained, freeze_backbone=freeze_backbone)
    if key == "slowfast":
        return SlowFastClassifier(pretrained=pretrained, freeze_backbone=freeze_backbone)
    if key == "swin3d":
        return TorchvisionSwin3DClassifier(pretrained=pretrained, freeze_backbone=freeze_backbone)
    if key == "josenet":
        if pretrained:
            raise RuntimeError(
                "JOSENet pretrained/self-supervised checkpoints are not configured in this repo. "
                "Run without --pretrained for a from-scratch comparison, or add explicit checkpoint paths first."
            )
        return OfficialJOSENetClassifier(freeze_backbone=freeze_backbone)
    raise ValueError(f"Unsupported model: {model_name}")


def model_status(model_name: str) -> str:
    key = model_name.lower().replace("-", "_")
    if key in UNAVAILABLE_OFFICIAL_MODELS:
        return "official_unavailable"
    return "available"


def validate_clip_length(model_name: str, clip_length: int) -> None:
    key = model_name.lower().replace("-", "_")
    allowed = ALLOWED_MODEL_CLIP_LENGTHS.get(key)
    if allowed is None:
        return
    if int(clip_length) not in allowed:
        allowed_text = ",".join(str(x) for x in sorted(allowed))
        raise ValueError(
            f"{model_name} does not support clip_length={clip_length} in this runner. "
            f"Allowed clip lengths: {allowed_text}."
        )


def move_batch_to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, (tuple, list)):
        return tuple(move_batch_to_device(item, device) for item in batch)
    return batch


def forward_model(model: nn.Module, inputs: Any) -> torch.Tensor:
    if isinstance(inputs, (tuple, list)):
        return model(inputs)
    return model(inputs)


def uses_binary_logits(model: nn.Module) -> bool:
    return bool(getattr(model, "binary_logits", False))


def compute_loss(criterion: nn.Module, logits: torch.Tensor, labels: torch.Tensor, binary_logits: bool) -> torch.Tensor:
    if binary_logits:
        return criterion(logits.flatten(), labels.float())
    return criterion(logits, labels)


def unpack_batch(batch: Any) -> tuple[Any, torch.Tensor, list[Any]]:
    if isinstance(batch, (tuple, list)) and len(batch) == 3:
        inputs, labels, metadata = batch
        return inputs, labels, list(metadata)
    inputs, labels = batch
    return inputs, labels, []


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
    criterion: nn.Module | None = None,
) -> Dict[str, Any]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    y_score: list[float] = []
    metadata_rows: list[dict[str, Any]] = []
    total_loss = 0.0
    binary_logits = uses_binary_logits(model)
    if criterion is None:
        criterion = nn.BCEWithLogitsLoss() if binary_logits else nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in tqdm(loader, desc="eval", leave=False):
            inputs, labels, metadata = unpack_batch(batch)
            inputs = move_batch_to_device(inputs, device)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", enabled=use_amp and device.type == "cuda"):
                logits = forward_model(model, inputs)
                loss = compute_loss(criterion, logits, labels, binary_logits)
            total_loss += float(loss.item()) * int(labels.numel())
            y_true.extend(labels.detach().cpu().tolist())
            if binary_logits:
                scores = torch.sigmoid(logits.flatten())
                y_pred.extend((scores >= 0.5).long().detach().cpu().tolist())
                y_score.extend(scores.detach().cpu().tolist())
            else:
                probs = torch.softmax(logits, dim=1)
                y_pred.extend(probs.argmax(dim=1).detach().cpu().tolist())
                y_score.extend(probs[:, 1].detach().cpu().tolist())
            for item in metadata:
                if isinstance(item, dict):
                    metadata_rows.append(
                        {
                            "video_id": str(item.get("video_id", "")),
                            "source_video": str(item.get("source_video", "")),
                            "source_dataset": str(item.get("source_dataset", "")),
                        }
                    )
                    continue
                metadata_rows.append(
                    {
                        "video_id": getattr(item, "video_id", ""),
                        "source_video": getattr(item, "source_video", ""),
                    }
                )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(x) for x in cm.ravel()]
    return {
        "loss": total_loss / max(len(y_true), 1),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_score": y_score,
        "metadata": metadata_rows,
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    criterion: nn.Module,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = False,
    grad_clip: float = 1.0,
) -> float:
    model.train()
    binary_logits = uses_binary_logits(model)
    total_loss = 0.0
    total_samples = 0
    for batch in tqdm(loader, desc="train", leave=False):
        inputs, labels, _metadata = unpack_batch(batch)
        inputs = move_batch_to_device(inputs, device)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None and use_amp and device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", enabled=True):
                logits = forward_model(model, inputs)
                loss = compute_loss(criterion, logits, labels, binary_logits)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = forward_model(model, inputs)
            loss = compute_loss(criterion, logits, labels, binary_logits)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
        total_loss += float(loss.item()) * int(labels.numel())
        total_samples += int(labels.numel())
    return total_loss / max(total_samples, 1)


def try_train_batch_size(
    model: nn.Module,
    dataset: Dataset,
    batch_size: int,
    device: torch.device,
    criterion: nn.Module,
    use_amp: bool,
    vram_target_mb: float,
) -> tuple[bool, float, str]:
    reset_peak_vram(device)
    loader = build_cached_dataloader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    try:
        batch = next(iter(loader))
        inputs, labels, _metadata = unpack_batch(batch)
        inputs = move_batch_to_device(inputs, device)
        labels = labels.to(device, non_blocking=True)
        model.train()
        model.zero_grad(set_to_none=True)
        binary_logits = uses_binary_logits(model)
        with torch.amp.autocast(device_type="cuda", enabled=use_amp and device.type == "cuda"):
            logits = forward_model(model, inputs)
            loss = compute_loss(criterion, logits, labels, binary_logits)
        loss.backward()
        peak = measure_peak_vram(device)
        model.zero_grad(set_to_none=True)
        ok = peak <= vram_target_mb
        reason = "" if ok else f"peak_vram_mb={peak:.0f} exceeds target={vram_target_mb:.0f}"
        del batch, inputs, labels, logits, loss
        torch.cuda.empty_cache()
        return ok, peak, reason
    except Exception as exc:
        model.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if is_oom_error(exc):
            return False, measure_peak_vram(device), "cuda_oom"
        raise


def auto_tune_batch_size(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    criterion: nn.Module,
    use_amp: bool,
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
        ok, peak, reason = try_train_batch_size(
            model, dataset, candidate, device, criterion, use_amp, target_mb
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
    if candidate > max_auto_batch and good == max_auto_batch:
        return good

    low = good + 1
    high = min(bad - 1, max_auto_batch)
    while low <= high:
        mid = (low + high) // 2
        ok, peak, reason = try_train_batch_size(
            model, dataset, mid, device, criterion, use_amp, target_mb
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


def write_predictions(path: Path, metrics: Dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["video_id", "source_video", "true_label", "pred_label", "score_violence"],
        )
        writer.writeheader()
        metadata = metrics.get("metadata", [])
        for idx, (label, pred, score) in enumerate(zip(metrics["y_true"], metrics["y_pred"], metrics["y_score"])):
            meta = metadata[idx] if idx < len(metadata) else {}
            writer.writerow(
                {
                    "video_id": meta.get("video_id", str(idx)),
                    "source_video": meta.get("source_video", ""),
                    "true_label": label,
                    "pred_label": pred,
                    "score_violence": score,
                }
            )


def strip_arrays(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in metrics.items() if k not in {"y_true", "y_pred", "y_score", "metadata"}}


def run(args: argparse.Namespace) -> Dict[str, Any]:
    set_runtime_optimizations(seed=args.seed)
    args.cache_root = require_cache_root(args.cache_root)
    if args.manifest is not None:
        args.manifest = require_existing_path(args.manifest, "manifest")
        manifest_header = pd.read_csv(args.manifest, nrows=1)
        required = {"cache_path", "video_id", "label", "split"}
        missing = sorted(required - set(manifest_header.columns))
        if missing:
            raise ValueError(f"Manifest is missing required columns: {missing}")
        manifest_splits = set(pd.read_csv(args.manifest, usecols=["split"])["split"].astype(str))
        if args.development_only and manifest_splits != {"train", "val"}:
            raise ValueError(
                f"Development-only manifest must contain exactly train/val; found {sorted(manifest_splits)}"
            )
    elif args.development_only:
        raise ValueError("--development-only requires an explicit --manifest")
    output_root = require_result_output(args.output_root)
    validate_clip_length(args.model, args.clip_length)
    if args.smoke:
        args.epochs = 1
        args.limit = args.limit or 8
        if str(args.batch_size).lower() == "auto":
            args.batch_size = "2"
        else:
            args.batch_size = min(int(args.batch_size), 2)
        args.pretrained = False
    key = args.model.lower().replace("-", "_")
    if key == "josenet" and int(args.num_workers) != 0:
        log.warning("JOSENet flow generation is CPU/OpenCV-heavy; forcing num_workers=0.")
        args.num_workers = 0
    defaults = MODEL_TRAIN_DEFAULTS.get(key, {})
    if args.lr is None:
        args.lr = float(defaults.get("lr", 1e-4))
    if args.weight_decay is None:
        args.weight_decay = float(defaults.get("weight_decay", 1e-4))
    if args.label_smoothing is None:
        args.label_smoothing = float(defaults.get("label_smoothing", 0.0))

    device = require_cuda_device(args.device)
    args.batch_size = parse_batch_size(args.batch_size)
    run_dir = output_root / args.model / f"t{args.clip_length}"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_ds = make_model_cache_dataset(
        args.model,
        "train",
        cache_root=args.cache_root,
        clip_length=args.clip_length,
        normalize=True,
        manifest_path=args.manifest,
    )
    val_ds = make_model_cache_dataset(
        args.model,
        "val",
        cache_root=args.cache_root,
        clip_length=args.clip_length,
        normalize=True,
        manifest_path=args.manifest,
    )
    test_ds = None
    if not args.development_only:
        test_ds = make_model_cache_dataset(
            args.model,
            "test",
            cache_root=args.cache_root,
            clip_length=args.clip_length,
            normalize=True,
            manifest_path=args.manifest,
        )
    train_ds = limit_dataset(train_ds, args.limit)
    val_ds = limit_dataset(val_ds, args.limit)
    if test_ds is not None:
        test_ds = limit_dataset(test_ds, args.limit)
    throughput_ds = val_ds if args.development_only else test_ds
    cache_read_throughput = measure_cache_read_throughput(
        throughput_ds, max_samples=args.throughput_samples
    )

    model = build_model(
        args.model,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
        clip_length=args.clip_length,
    ).to(device)
    model_implementation = str(getattr(model, "implementation", model.__class__.__name__))
    class_weights = make_class_weights(train_ds, device)
    binary_logits = uses_binary_logits(model)
    if binary_logits:
        pos_weight = (class_weights[1] / class_weights[0]).detach()
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)

    if args.batch_size == "auto":
        args.batch_size = auto_tune_batch_size(
            model,
            train_ds,
            device,
            criterion,
            use_amp=args.amp,
            vram_target=args.vram_target,
            max_auto_batch=args.max_auto_batch,
        )

    train_loader = build_cached_dataloader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = build_cached_dataloader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = None
    if test_ds is not None:
        test_loader = build_cached_dataloader(
            test_ds,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    trainable_parameters = [param for param in model.parameters() if param.requires_grad]
    if key == "josenet":
        optimizer = torch.optim.SGD(
            trainable_parameters,
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
            nesterov=True,
        )
    else:
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = torch.amp.GradScaler("cuda") if args.amp and device.type == "cuda" else None

    history: list[Dict[str, Any]] = []
    best_score = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    start_time = time.time()
    best_path = run_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        reset_peak_vram(device)
        epoch_start = time.time()
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            criterion,
            scaler=scaler,
            use_amp=args.amp,
            grad_clip=args.grad_clip,
        )
        val_metrics = evaluate(model, val_loader, device, use_amp=args.amp, criterion=criterion)
        epoch_seconds = time.time() - epoch_start
        peak_vram_mb = measure_peak_vram(device)
        scheduler.step()
        if args.selection_metric == "balanced":
            selection_score = 0.5 * val_metrics["balanced_accuracy"] + 0.5 * val_metrics["f1_macro"]
        else:
            selection_score = val_metrics["f1"]
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "epoch_seconds": epoch_seconds,
            "peak_vram_mb": peak_vram_mb,
            "selection_score": selection_score,
            **strip_arrays(val_metrics),
        }
        history.append(row)
        log.info(
            "epoch=%d train_loss=%.4f val_score=%.4f val_f1=%.4f val_bal_acc=%.4f val_f1_macro=%.4f peak_vram=%.0fMB time=%.1fs",
            epoch,
            train_loss,
            selection_score,
            val_metrics["f1"],
            val_metrics["balanced_accuracy"],
            val_metrics["f1_macro"],
            peak_vram_mb,
            epoch_seconds,
        )
        if selection_score > best_score:
            best_score = selection_score
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "model_name": args.model,
                    "clip_length": args.clip_length,
                    "cache_root": str(args.cache_root),
                    "selection_metric": args.selection_metric,
                    "selection_score": selection_score,
                    "metrics": strip_arrays(val_metrics),
                },
                best_path,
            )
        else:
            epochs_without_improvement += 1
            if args.patience > 0 and epochs_without_improvement >= args.patience:
                log.info(
                    "early stopping at epoch=%d best_epoch=%d best_score=%.4f patience=%d",
                    epoch,
                    best_epoch,
                    best_score,
                    args.patience,
                )
                break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    validation_metrics = evaluate(model, val_loader, device, use_amp=args.amp, criterion=criterion)
    write_predictions(run_dir / "validation_predictions.csv", validation_metrics)
    (run_dir / "validation_metrics.json").write_text(
        json.dumps(strip_arrays(validation_metrics), indent=2), encoding="utf-8"
    )

    common_summary = {
        "model": args.model,
        "model_status": model_status(args.model),
        "model_implementation": model_implementation,
        "clip_length": args.clip_length,
        "cache_root": str(args.cache_root),
        "manifest": str(args.manifest) if args.manifest is not None else None,
        "manifest_sha256": sha256_file(args.manifest) if args.manifest is not None else None,
        "output_dir": str(run_dir),
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "vram_target": args.vram_target,
        "max_auto_batch": args.max_auto_batch,
        "limit": args.limit,
        "development_only": bool(args.development_only),
        "pretrained": args.pretrained,
        "freeze_backbone": args.freeze_backbone,
        "amp": args.amp,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "loss": "bce_with_logits" if binary_logits else "cross_entropy",
        "optimizer": "sgd_momentum_nesterov" if key == "josenet" else "adamw",
        "grad_clip": args.grad_clip,
        "selection_metric": args.selection_metric,
        "seed": args.seed,
        "model_input_frames": 16 if key == "josenet" else args.clip_length,
        "class_weights": [float(x) for x in class_weights.detach().cpu().tolist()],
        "elapsed_seconds": time.time() - start_time,
        "cache_read_throughput": cache_read_throughput,
        "peak_vram_mb": float(max([float(row.get("peak_vram_mb", 0.0)) for row in history] or [0.0])),
        "best_epoch": int(checkpoint.get("epoch", best_epoch)),
        "best_val_score": best_score,
        "best_val_f1": float(checkpoint.get("metrics", {}).get("f1", 0.0)),
        "best_val_balanced_accuracy": float(checkpoint.get("metrics", {}).get("balanced_accuracy", 0.0)),
        "best_val_f1_macro": float(checkpoint.get("metrics", {}).get("f1_macro", 0.0)),
        "validation": strip_arrays(validation_metrics),
    }
    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(history[0].keys()) if history else ["epoch"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
    if args.development_only:
        with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(common_summary, handle, indent=2)
        log.info("Development-only experiment complete; test dataset was never constructed.")
        return common_summary

    if test_loader is None:
        raise RuntimeError("Test evaluation requested without a test dataset")
    test_metrics = evaluate(model, test_loader, device, use_amp=args.amp, criterion=criterion)
    elapsed = time.time() - start_time

    write_predictions(run_dir / "predictions.csv", test_metrics)

    summary = dict(common_summary)
    summary["elapsed_seconds"] = elapsed
    summary["test"] = strip_arrays(test_metrics)
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run JRTIP experiments from result/gpu_cache.")
    parser.add_argument(
        "--model",
        choices=["c3d", "i3d", "resnet_lstm", "slowfast", "swin3d", "josenet"],
        required=True,
    )
    parser.add_argument("--clip-length", type=int, choices=[16, 32, 64], default=16)
    parser.add_argument("--cache-root", type=Path, default=Path("result/gpu_cache"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("result/cached_experiments"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", default="4", help="Positive integer or 'auto'.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--label-smoothing", type=float, default=None)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--throughput-samples", type=int, default=16)
    parser.add_argument("--vram-target", type=float, default=0.92)
    parser.add_argument("--max-auto-batch", type=int, default=128)
    parser.add_argument(
        "--selection-metric",
        choices=["balanced", "f1"],
        default="balanced",
        help="balanced uses 0.5*balanced_accuracy + 0.5*f1_macro to avoid one-class checkpoints.",
    )
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--development-only",
        action="store_true",
        help="Train/evaluate validation only; requires an explicit train/val manifest.",
    )
    return parser.parse_args(argv)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    args = parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
