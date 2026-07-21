"""
Real inference benchmark runner for the JRTIP paper.

The runner intentionally reads only the locked cache under result/gpu_cache for
classifier-only timing. It loads trained checkpoints from result/, measures
Batch=1 FP16 inference latency over 3 runs, and writes both summary and raw
latency files under result/inference_benchmark/.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MOVINET_PATH = REPO_ROOT / "training_code" / "train_movinet_violence" / "MoViNet-pytorch"
if str(MOVINET_PATH) not in sys.path:
    sys.path.insert(0, str(MOVINET_PATH))

from data.processors.gpu_clip_cache_dataset import GpuClipCacheDataset, assert_new_cache_path
from data.processors.model_cache_adapters import make_model_cache_dataset
from training_code.run_jrtip_cached_experiments import build_model as build_baseline_model
from training_code.run_jrtip_cached_experiments import forward_model, move_batch_to_device
from training_code.run_movinet_cached_experiments import create_model as create_movinet_model


SUMMARY_FIELDS = [
    "model",
    "clip_length",
    "status",
    "accuracy",
    "precision",
    "recall",
    "checkpoint",
    "checkpoint_source",
    "fps_batch1",
    "latency_mean_ms",
    "latency_median_ms",
    "latency_p95_ms",
    "gflops",
    "gflops_tool",
    "params_M",
    "peak_vram_mb",
    "power_w",
    "warmup_iters",
    "measured_iters",
    "runs",
    "input_shape",
    "inference_precision",
    "device",
    "cache_root",
    "error",
]

PIPELINE_FIELDS = [
    "model",
    "clip_length",
    "status",
    "videos",
    "frames",
    "movinet_calls",
    "yolo_ms",
    "bytetrack_ms",
    "hdbscan_ms",
    "gate_ms",
    "movinet_ms",
    "total_ms",
    "video_paths",
    "error",
]


@dataclass(frozen=True)
class BenchmarkJob:
    model: str
    kind: str
    windows: tuple[int, ...]


JOBS: tuple[BenchmarkJob, ...] = (
    BenchmarkJob("MoViNet_M1", "movinet", (50,)),
    BenchmarkJob("MoViNet_M2", "movinet", (50,)),
    BenchmarkJob("MoViNet_M3", "movinet", (50,)),
    BenchmarkJob("C3D", "baseline", (16,)),
    BenchmarkJob("I3D", "baseline", (32,)),
    BenchmarkJob("ResNet-LSTM", "baseline", (32,)),
    BenchmarkJob("SlowFast", "baseline", (32,)),
    BenchmarkJob("Swin3D", "baseline", (32,)),
    BenchmarkJob("JOSENet", "baseline", (16,)),
)

BASELINE_KEYS = {
    "C3D": "c3d",
    "I3D": "i3d",
    "ResNet-LSTM": "resnet_lstm",
    "SlowFast": "slowfast",
    "Swin3D": "swin3d",
    "JOSENet": "josenet",
}

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1, 1)


def require_result_path(path: str | Path, field_name: str) -> Path:
    p = Path(path)
    parts = [part.lower() for part in p.parts]
    if "result" not in parts and (not parts or parts[0].lower() != "result"):
        raise ValueError(f"{field_name} must be under result/: {path}")
    return p


def require_new_cache_root(path: str | Path) -> Path:
    assert_new_cache_path(path, "cache_root")
    p = Path(path)
    parts = [part.lower() for part in p.parts]
    ok = any(parts[i] == "result" and i + 1 < len(parts) and parts[i + 1] == "gpu_cache" for i in range(len(parts)))
    if not ok:
        raise ValueError(f"cache_root must point to the new cache under result/gpu_cache: {path}")
    return p


def set_runtime() -> None:
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def device_from_arg(raw: str) -> torch.device:
    if raw == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(raw)


def checkpoint_for(model: str, clip_length: int, movinet_root: Path | None = None) -> tuple[Path, str]:
    if model.startswith("MoViNet_"):
        variant = model.rsplit("_", 1)[-1]
        source_root = movinet_root or (REPO_ROOT / "result" / "movinet_cached_experiments")
        root = source_root / f"variant_{variant}" / f"t{clip_length}"
        source = source_root.relative_to(REPO_ROOT).as_posix() if source_root.is_absolute() else source_root.as_posix()
        return root / "best.pt", source
    if model == "JOSENet":
        candidates = [
            REPO_ROOT / "result" / "cached_experiments_josenet_official" / "josenet" / f"t{clip_length}" / "best.pt",
            REPO_ROOT / "result" / "cached_experiments" / "josenet" / f"t{clip_length}" / "best.pt",
        ]
        for candidate in candidates:
            if candidate.exists():
                source = candidate.parents[2].relative_to(REPO_ROOT).as_posix()
                return candidate, source
        return candidates[0], "result/cached_experiments_josenet_official + result/cached_experiments"
    if model in BASELINE_KEYS:
        key = BASELINE_KEYS[model]
        candidates = [
            REPO_ROOT / "result" / "cached_experiments" / key / f"t{clip_length}" / "best.pt",
            REPO_ROOT / "result" / "cached_experiments_true_arch_pretrained" / key / f"t{clip_length}" / "best.pt",
            REPO_ROOT / "result" / "cached_experiments_true_arch_rerun" / key / f"t{clip_length}" / "best.pt",
            REPO_ROOT / "result" / "cached_experiments_true_arch_scratch" / key / f"t{clip_length}" / "best.pt",
        ]
        for candidate in candidates:
            if candidate.exists():
                source = candidate.parents[2].relative_to(REPO_ROOT).as_posix()
                return candidate, source
        return candidates[0], "result/cached_experiments + legacy cached_experiments_true_arch_*"
    raise ValueError(f"Unsupported model: {model}")


def load_state(model: nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {checkpoint_path}")
    if "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    elif "model" in checkpoint:
        state = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    else:
        raise RuntimeError(f"No model state found in checkpoint: {checkpoint_path}")
    model.load_state_dict(state)


def load_classification_metrics(checkpoint_path: Path) -> dict[str, Any]:
    metrics_path = checkpoint_path.parent / "metrics.json"
    if not metrics_path.exists():
        return {"accuracy": "", "precision": "", "recall": ""}
    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        test = data.get("test", data)
        return {
            "accuracy": test.get("accuracy", ""),
            "precision": test.get("precision", ""),
            "recall": test.get("recall", ""),
        }
    except Exception as exc:
        return {
            "accuracy": "",
            "precision": "",
            "recall": "",
            "metrics_error": f"{type(exc).__name__}: {exc}",
        }


def build_model_and_sample(model_name: str, clip_length: int, cache_root: Path, device: torch.device) -> tuple[nn.Module, Any]:
    if model_name.startswith("MoViNet_"):
        model = create_movinet_model(device)
        preprocess_type = "movinet_preprocessed" if model_name == "MoViNet_M3" else "wholeframe"
        dataset = GpuClipCacheDataset(
            cache_root=cache_root,
            clip_length=clip_length,
            preprocess_type=preprocess_type,
            size=224,
            split="test",
            normalize=False,
        )
        sample, _label, _metadata = dataset[0]
        return model, sample.unsqueeze(0)

    key = BASELINE_KEYS[model_name]
    model = build_baseline_model(key, pretrained=False, freeze_backbone=False, clip_length=clip_length).to(device)
    dataset = make_model_cache_dataset(key, "test", cache_root=cache_root, clip_length=clip_length, normalize=True)
    sample, _label = dataset[0]
    if isinstance(sample, (tuple, list)):
        return model, tuple(x.unsqueeze(0) for x in sample)
    return model, sample.unsqueeze(0)


class ForwardWrapper(nn.Module):
    def __init__(self, model: nn.Module, tuple_input: bool) -> None:
        super().__init__()
        self.model = model
        self.tuple_input = tuple_input

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, "clean_activation_buffers"):
            self.model.clean_activation_buffers()
        inputs: Any = tuple(args) if self.tuple_input else args[0]
        out = forward_model(self.model, inputs)
        if hasattr(self.model, "clean_activation_buffers"):
            self.model.clean_activation_buffers()
        return out


def flatten_inputs(sample: Any) -> tuple[tuple[torch.Tensor, ...], bool]:
    if isinstance(sample, (tuple, list)):
        return tuple(sample), True
    return (sample,), False


def input_shape(sample: Any) -> str:
    if isinstance(sample, (tuple, list)):
        return " + ".join(str(tuple(x.shape)) for x in sample)
    return str(tuple(sample.shape))


def count_params_m(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def measure_gflops(wrapper: nn.Module, flat_inputs: tuple[torch.Tensor, ...]) -> tuple[float | str, str]:
    try:
        from fvcore.nn import FlopCountAnalysis

        flops = FlopCountAnalysis(wrapper, flat_inputs)
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        flops.tracer_warnings("none")
        total = float(flops.total()) / 1e9
        return total, "fvcore.nn.FlopCountAnalysis"
    except Exception as exc:
        return "not_available", f"fvcore_failed:{type(exc).__name__}"


class PowerSampler:
    def __init__(self, interval_s: float = 0.1) -> None:
        self.interval_s = float(interval_s)
        self.samples: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> list[float]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        return self.samples

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                proc = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                if proc.returncode == 0:
                    first = proc.stdout.strip().splitlines()[0].strip()
                    self.samples.append(float(first))
            except Exception:
                pass
            self._stop.wait(self.interval_s)


def cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_raw_run(path: Path, model: str, clip_length: int, run_idx: int, latencies: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "clip_length", "run", "iteration", "latency_ms"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for idx, latency in enumerate(latencies, start=1):
            writer.writerow(
                {
                    "model": model,
                    "clip_length": clip_length,
                    "run": run_idx,
                    "iteration": idx,
                    "latency_ms": latency,
                }
            )


def benchmark_one(
    model_name: str,
    clip_length: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint_path, source = checkpoint_for(model_name, clip_length, args.movinet_root)
    row: dict[str, Any] = {
        "model": model_name,
        "clip_length": clip_length,
        "status": "failed",
        **load_classification_metrics(checkpoint_path),
        "checkpoint": str(checkpoint_path.relative_to(REPO_ROOT)) if checkpoint_path.is_absolute() else str(checkpoint_path),
        "checkpoint_source": source,
        "warmup_iters": args.warmup_iters,
        "measured_iters": args.measured_iters,
        "runs": args.runs,
        "inference_precision": "fp16_autocast",
        "device": str(device),
        "cache_root": str(args.cache_root),
        "error": "",
    }

    model, sample = build_model_and_sample(model_name, clip_length, args.cache_root, device)
    load_state(model, checkpoint_path, device)
    model.eval()
    sample = move_batch_to_device(sample, device)
    flat_inputs, tuple_input = flatten_inputs(sample)
    wrapper = ForwardWrapper(model, tuple_input).to(device).eval()

    row["input_shape"] = input_shape(sample)
    row["params_M"] = count_params_m(model)
    row["gflops"], row["gflops_tool"] = measure_gflops(wrapper, flat_inputs)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    all_latencies: list[float] = []
    run_fps: list[float] = []
    run_power: list[float] = []
    run_peak_vram: list[float] = []

    with torch.inference_mode():
        for run_idx in range(1, int(args.runs) + 1):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            for _ in range(int(args.warmup_iters)):
                with torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda"):
                    _ = wrapper(*flat_inputs)
            cuda_sync(device)

            sampler = PowerSampler(interval_s=args.power_interval_s)
            sampler.start()
            latencies: list[float] = []
            for _ in range(int(args.measured_iters)):
                start = time.perf_counter()
                with torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda"):
                    _ = wrapper(*flat_inputs)
                cuda_sync(device)
                latencies.append((time.perf_counter() - start) * 1000.0)
            samples = sampler.stop()

            raw_path = args.raw_dir / f"{model_name}_t{clip_length}_run{run_idx}.csv"
            write_raw_run(raw_path, model_name, clip_length, run_idx, latencies)

            total_s = max(sum(latencies) / 1000.0, 1e-12)
            run_fps.append(float(len(latencies)) / total_s)
            all_latencies.extend(latencies)
            if samples:
                run_power.append(float(statistics.mean(samples)))
            if device.type == "cuda":
                run_peak_vram.append(float(torch.cuda.max_memory_allocated(device) / (1024**2)))

    row.update(
        {
            "status": "ok",
            "fps_batch1": float(statistics.mean(run_fps)),
            "latency_mean_ms": float(statistics.mean(all_latencies)),
            "latency_median_ms": float(statistics.median(all_latencies)),
            "latency_p95_ms": float(np.percentile(all_latencies, 95)),
            "peak_vram_mb": float(max(run_peak_vram)) if run_peak_vram else "not_available",
            "power_w": float(statistics.mean(run_power)) if run_power else "not_available",
        }
    )
    return row


def normalize_frame(frame_bgr: np.ndarray) -> torch.Tensor:
    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(frame).permute(2, 0, 1).float().div_(255.0)
    return (tensor.view(3, 1, 224, 224) - IMAGENET_MEAN).div_(IMAGENET_STD).squeeze(1)


def bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return float(inter / max(area_a + area_b - inter, 1))


def largest_cluster_bbox(boxes: np.ndarray, labels: np.ndarray | None, frame_shape: tuple[int, int, int]) -> tuple[int, int, int, int] | None:
    if boxes.size == 0:
        return None
    if labels is None:
        selected = boxes
    else:
        cluster_ids = [int(x) for x in set(labels.tolist()) if int(x) != -1]
        if not cluster_ids:
            selected = boxes
        else:
            best = max(cluster_ids, key=lambda cid: int(np.sum(labels == cid)))
            selected = boxes[labels == best]
    h, w = frame_shape[:2]
    x1 = int(max(0, np.min(selected[:, 0])))
    y1 = int(max(0, np.min(selected[:, 1])))
    x2 = int(min(w - 1, np.max(selected[:, 2])))
    y2 = int(min(h - 1, np.max(selected[:, 3])))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def discover_pipeline_videos(root: Path, limit: int) -> list[Path]:
    exts = {".mp4", ".avi", ".mov", ".mkv", ".flv"}
    violent = sorted([p for p in (root / "violence").rglob("*") if p.suffix.lower() in exts]) if (root / "violence").exists() else []
    normal = sorted([p for p in (root / "non_violence").rglob("*") if p.suffix.lower() in exts]) if (root / "non_violence").exists() else []

    def frame_count(path: Path) -> int:
        cap = cv2.VideoCapture(str(path))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return count

    videos = sorted(violent, key=frame_count, reverse=True) + sorted(normal, key=frame_count, reverse=True)
    return videos[: int(limit)]


def benchmark_m3_pipeline(args: argparse.Namespace, device: torch.device) -> list[dict[str, Any]]:
    try:
        import hdbscan
        from ultralytics import YOLO
    except Exception as exc:
        return [
            {
                "model": "MoViNet_M3_pipeline",
                "clip_length": "",
                "status": "failed",
                "error": f"pipeline dependencies unavailable: {type(exc).__name__}: {exc}",
            }
        ]

    video_root = Path(args.pipeline_video_root)
    videos = discover_pipeline_videos(video_root, args.pipeline_video_limit)
    if not videos:
        return [
            {
                "model": "MoViNet_M3_pipeline",
                "clip_length": "",
                "status": "failed",
                "error": f"no videos found under {video_root}",
            }
        ]

    yolo = YOLO(str(args.yolo_model))
    rows: list[dict[str, Any]] = []
    for clip_length in args.pipeline_clip_lengths:
        row: dict[str, Any] = {
            "model": "MoViNet_M3_pipeline",
            "clip_length": int(clip_length),
            "status": "failed",
            "videos": len(videos),
            "video_paths": json.dumps([str(p) for p in videos], ensure_ascii=False),
            "error": "",
        }
        try:
            model, sample = build_model_and_sample("MoViNet_M3", int(clip_length), args.cache_root, device)
            checkpoint_path, _source = checkpoint_for("MoViNet_M3", int(clip_length), args.movinet_root)
            load_state(model, checkpoint_path, device)
            model.eval()
            sample = move_batch_to_device(sample, device)
            for _ in range(min(10, int(args.warmup_iters))):
                with torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda"):
                    _ = model(sample)
                cuda_sync(device)
                if hasattr(model, "clean_activation_buffers"):
                    model.clean_activation_buffers()

            yolo_ms: list[float] = []
            bytetrack_ms: list[float] = []
            hdbscan_ms: list[float] = []
            gate_ms: list[float] = []
            movinet_ms: list[float] = []
            frame_total_ms: list[float] = []
            total_frames = 0
            movinet_calls = 0

            with torch.inference_mode():
                for video in videos:
                    cap = cv2.VideoCapture(str(video))
                    if not cap.isOpened():
                        continue
                    prev_bbox: tuple[int, int, int, int] | None = None
                    clip_buffer: list[torch.Tensor] = []
                    active_frames_remaining = 0
                    frames_seen = 0
                    while frames_seen < int(args.pipeline_max_frames):
                        ok, frame = cap.read()
                        if not ok:
                            break
                        frames_seen += 1
                        total_frames += 1
                        frame_start = time.perf_counter()

                        t0 = time.perf_counter()
                        results = yolo.track(
                            frame,
                            classes=[0],
                            conf=float(args.yolo_conf),
                            tracker="bytetrack.yaml",
                            persist=True,
                            verbose=False,
                        )
                        track_total = (time.perf_counter() - t0) * 1000.0
                        speed = getattr(results[0], "speed", {}) if results else {}
                        yolo_speed = float(speed.get("preprocess", 0.0) + speed.get("inference", 0.0) + speed.get("postprocess", 0.0))
                        yolo_ms.append(yolo_speed if yolo_speed > 0 else track_total)
                        bytetrack_ms.append(max(0.0, track_total - yolo_speed))

                        boxes = np.empty((0, 4), dtype=np.float32)
                        if results and results[0].boxes is not None and results[0].boxes.xyxy is not None:
                            boxes = results[0].boxes.xyxy.detach().cpu().numpy().astype(np.float32)
                        centers = np.column_stack(((boxes[:, 0] + boxes[:, 2]) / 2.0, (boxes[:, 1] + boxes[:, 3]) / 2.0)) if len(boxes) else np.empty((0, 2))

                        t0 = time.perf_counter()
                        cluster_labels = None
                        if len(centers) >= 2:
                            cluster_labels = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=2).fit_predict(centers)
                        hdbscan_ms.append((time.perf_counter() - t0) * 1000.0)

                        t0 = time.perf_counter()
                        bbox = largest_cluster_bbox(boxes, cluster_labels, frame.shape)
                        gate_open = False
                        if bbox is not None:
                            if prev_bbox is None:
                                gate_open = True
                            else:
                                iou = bbox_iou(prev_bbox, bbox)
                                cx = (bbox[0] + bbox[2]) * 0.5
                                cy = (bbox[1] + bbox[3]) * 0.5
                                pcx = (prev_bbox[0] + prev_bbox[2]) * 0.5
                                pcy = (prev_bbox[1] + prev_bbox[3]) * 0.5
                                velocity = math.hypot(cx - pcx, cy - pcy) / max(frame.shape[0], frame.shape[1])
                                gate_open = iou <= float(args.iou_gate) or velocity >= float(args.velocity_gate)
                            prev_bbox = bbox
                        gate_ms.append((time.perf_counter() - t0) * 1000.0)

                        if gate_open and bbox is not None:
                            active_frames_remaining = int(clip_length)
                            clip_buffer.clear()

                        if active_frames_remaining > 0 and bbox is not None:
                            x1, y1, x2, y2 = bbox
                            crop = frame[y1:y2, x1:x2]
                            if crop.size > 0:
                                clip_buffer.append(normalize_frame(crop))
                                active_frames_remaining -= 1
                                if len(clip_buffer) > int(clip_length):
                                    clip_buffer = clip_buffer[-int(clip_length) :]
                                if len(clip_buffer) == int(clip_length):
                                    clip = torch.stack(clip_buffer, dim=1).unsqueeze(0).to(device, non_blocking=True)
                                    t0 = time.perf_counter()
                                    with torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda"):
                                        _ = model(clip)
                                    cuda_sync(device)
                                    if hasattr(model, "clean_activation_buffers"):
                                        model.clean_activation_buffers()
                                    movinet_ms.append((time.perf_counter() - t0) * 1000.0)
                                    movinet_calls += 1

                        frame_total_ms.append((time.perf_counter() - frame_start) * 1000.0)
                    cap.release()

            row.update(
                {
                    "status": "ok" if total_frames else "failed",
                    "frames": total_frames,
                    "movinet_calls": movinet_calls,
                    "yolo_ms": float(statistics.mean(yolo_ms)) if yolo_ms else "",
                    "bytetrack_ms": float(statistics.mean(bytetrack_ms)) if bytetrack_ms else "",
                    "hdbscan_ms": float(statistics.mean(hdbscan_ms)) if hdbscan_ms else "",
                    "gate_ms": float(statistics.mean(gate_ms)) if gate_ms else "",
                    "movinet_ms": float(statistics.mean(movinet_ms)) if movinet_ms else "",
                    "total_ms": float(statistics.mean(frame_total_ms)) if frame_total_ms else "",
                    "error": "" if total_frames else "no frames processed",
                }
            )
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return rows


def parse_csv_list(raw: str | None) -> set[str] | None:
    if raw is None or raw.strip().lower() == "all":
        return None
    return {x.strip() for x in raw.split(",") if x.strip()}


def selected_jobs(args: argparse.Namespace) -> list[tuple[str, int]]:
    model_filter = parse_csv_list(args.models)
    clip_filter = {int(x) for x in args.clip_lengths} if args.clip_lengths else None
    pairs: list[tuple[str, int]] = []
    for job in JOBS:
        if model_filter is not None and job.model not in model_filter:
            continue
        for window in job.windows:
            if clip_filter is not None and int(window) not in clip_filter:
                continue
            pairs.append((job.model, int(window)))
    return pairs


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real JRTIP inference benchmarks.")
    parser.add_argument("--cache-root", type=Path, default=Path("result/gpu_cache"))
    parser.add_argument("--movinet-root", type=Path, default=Path("result/movinet_cached_experiments"))
    parser.add_argument("--output-csv", type=Path, default=Path("result/inference_benchmark/inference_summary.csv"))
    parser.add_argument("--raw-dir", type=Path, default=Path("result/inference_benchmark/raw_runs"))
    parser.add_argument("--pipeline-output-csv", type=Path, default=Path("result/inference_benchmark/pipeline_breakdown.csv"))
    parser.add_argument("--models", default="all", help="Comma-separated model names or all.")
    parser.add_argument("--clip-lengths", nargs="*", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup-iters", type=int, default=50)
    parser.add_argument("--measured-iters", type=int, default=100)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--power-interval-s", type=float, default=0.1)
    parser.add_argument("--continue-on-error", action="store_true", default=True)
    parser.add_argument("--skip-pipeline", action="store_true")
    parser.add_argument("--pipeline-clip-lengths", nargs="*", type=int, default=[16, 32, 64])
    parser.add_argument("--pipeline-video-root", type=Path, default=Path("data/merged/test_videos"))
    parser.add_argument("--pipeline-video-limit", type=int, default=3)
    parser.add_argument("--pipeline-max-frames", type=int, default=128)
    parser.add_argument("--yolo-model", type=Path, default=Path("yolo11n.pt"))
    parser.add_argument("--yolo-conf", type=float, default=0.2)
    parser.add_argument("--iou-gate", type=float, default=0.85)
    parser.add_argument("--velocity-gate", type=float, default=0.05)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    args.cache_root = require_new_cache_root(args.cache_root)
    args.movinet_root = require_result_path(args.movinet_root, "movinet_root")
    args.output_csv = require_result_path(args.output_csv, "output_csv")
    args.raw_dir = require_result_path(args.raw_dir, "raw_dir")
    args.pipeline_output_csv = require_result_path(args.pipeline_output_csv, "pipeline_output_csv")
    set_runtime()
    device = device_from_arg(args.device)

    rows: list[dict[str, Any]] = []
    pairs = selected_jobs(args)
    print(f"Benchmarking {len(pairs)} classifier jobs on {device}", flush=True)
    for model_name, clip_length in pairs:
        print(f"[classifier] {model_name} t{clip_length}", flush=True)
        try:
            row = benchmark_one(model_name, clip_length, args, device)
        except Exception as exc:
            checkpoint_path, source = checkpoint_for(model_name, clip_length, args.movinet_root)
            row = {
                "model": model_name,
                "clip_length": clip_length,
                "status": "failed",
                **load_classification_metrics(checkpoint_path),
                "checkpoint": str(checkpoint_path),
                "checkpoint_source": source,
                "warmup_iters": args.warmup_iters,
                "measured_iters": args.measured_iters,
                "runs": args.runs,
                "inference_precision": "fp16_autocast",
                "device": str(device),
                "cache_root": str(args.cache_root),
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"  failed: {row['error']}", flush=True)
            if not args.continue_on_error:
                rows.append(row)
                write_csv(args.output_csv, rows, SUMMARY_FIELDS)
                raise
        rows.append(row)
        write_csv(args.output_csv, rows, SUMMARY_FIELDS)

    if not args.skip_pipeline:
        print("[pipeline] MoViNet_M3 full pipeline", flush=True)
        pipeline_rows = benchmark_m3_pipeline(args, device)
        write_csv(args.pipeline_output_csv, pipeline_rows, PIPELINE_FIELDS)

    if pairs:
        print(f"Wrote {args.output_csv}", flush=True)
    else:
        print("Classifier benchmark skipped; existing summary CSV was left unchanged.", flush=True)
    if not args.skip_pipeline:
        print(f"Wrote {args.pipeline_output_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
