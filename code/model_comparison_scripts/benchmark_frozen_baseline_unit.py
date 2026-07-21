#!/usr/bin/env python3
"""Benchmark one frozen baseline checkpoint on 100 common test inference units.

Disk/cache decoding and architecture-specific CPU preprocessing occur outside
the timed interval.  The primary latency is synchronized batch-one model-core
forward latency; this is intentionally distinct from the existing 2K streaming
system benchmark.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.unsqueeze(0).to(device, non_blocking=False)
    if isinstance(value, (tuple, list)):
        return tuple(move_to_device(item, device) for item in value)
    raise TypeError(f"Unsupported input type: {type(value)}")


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--clip-length", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--benchmark-ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat-index", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--units", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.output.exists():
        raise RuntimeError(f"Refusing to overwrite benchmark output: {args.output}")
    if sha256_file(args.checkpoint) != args.checkpoint_sha256:
        raise RuntimeError("Checkpoint hash mismatch")
    if sha256_file(args.manifest) != args.manifest_sha256:
        raise RuntimeError("Manifest hash mismatch")
    with args.benchmark_ids.open(newline="", encoding="utf-8-sig") as handle:
        id_rows = list(csv.DictReader(handle))
    target_ids = [row["video_id"] for row in id_rows]
    if len(target_ids) != args.units or not target_ids:
        raise RuntimeError("Benchmark ID file must contain exactly --units rows")

    repo = args.repo_root.resolve()
    sys.path.insert(0, str(repo))
    from data.processors.model_cache_adapters import make_model_cache_dataset
    from training_code.run_jrtip_cached_experiments import build_model, forward_model

    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA is required for this benchmark")
    device = torch.device(args.device)
    gpu_name = torch.cuda.get_device_name(device)
    summary: dict[str, Any] = {
        "status": "started",
        "created_utc": now(),
        "model_id": args.model_id,
        "repeat_index": args.repeat_index,
        "warmup_units": args.warmup,
        "measured_units": args.units,
        "timing_scope": "model_core_forward_batch1_preprocessed_input",
        "checkpoint_sha256": args.checkpoint_sha256,
        "manifest_sha256": args.manifest_sha256,
        "benchmark_ids_sha256": sha256_file(args.benchmark_ids),
        "unique_benchmark_ids": len(set(target_ids)),
        "gpu_name": gpu_name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    atomic_json(args.output, summary)

    try:
        dataset = make_model_cache_dataset(
            args.model_id,
            "test",
            cache_root=args.cache_root.resolve(),
            clip_length=args.clip_length,
            normalize=True,
            manifest_path=args.manifest,
        )
        base_manifest = dataset.base.manifest
        id_to_index = {str(row.video_id): int(index) for index, row in base_manifest.reset_index(drop=True).iterrows()}
        missing = [video_id for video_id in target_ids if video_id not in id_to_index]
        if missing:
            raise RuntimeError(f"Benchmark IDs absent from dataset: {missing[:3]}")

        model = build_model(args.model_id, pretrained=False, freeze_backbone=False, clip_length=args.clip_length).to(device)
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if str(checkpoint.get("model_name")) != args.model_id or int(checkpoint.get("clip_length", -1)) != args.clip_length:
            raise RuntimeError("Checkpoint identity mismatch")
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        torch.cuda.reset_peak_memory_stats(device)

        warmup_ids = target_ids[: max(1, min(args.warmup, len(target_ids)))]
        with torch.inference_mode():
            for video_id in warmup_ids:
                inputs, _label, metadata = dataset[id_to_index[video_id]]
                if str(metadata["video_id"]) != video_id:
                    raise RuntimeError("Dataset benchmark ID alignment failure")
                gpu_inputs = move_to_device(inputs, device)
                with torch.amp.autocast(device_type="cuda", enabled=True):
                    _ = forward_model(model, gpu_inputs)
                torch.cuda.synchronize(device)
                del gpu_inputs, inputs

        rows: list[dict[str, Any]] = []
        with torch.inference_mode():
            for unit_index, video_id in enumerate(target_ids, start=1):
                inputs, label, metadata = dataset[id_to_index[video_id]]
                if str(metadata["video_id"]) != video_id:
                    raise RuntimeError("Dataset benchmark ID alignment failure")
                gpu_inputs = move_to_device(inputs, device)
                torch.cuda.synchronize(device)
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                wall_start = time.perf_counter()
                start_event.record()
                with torch.amp.autocast(device_type="cuda", enabled=True):
                    output = forward_model(model, gpu_inputs)
                end_event.record()
                torch.cuda.synchronize(device)
                wall_ms = (time.perf_counter() - wall_start) * 1000.0
                gpu_ms = float(start_event.elapsed_time(end_event))
                rows.append({
                    "unit_index": unit_index,
                    "video_id": video_id,
                    "true_label": int(label),
                    "wall_latency_ms": wall_ms,
                    "gpu_latency_ms": gpu_ms,
                    "output_elements": int(output.numel()),
                })
                del gpu_inputs, inputs, output

        raw_path = args.output.with_name(args.output.stem + "_raw.csv")
        write_csv(raw_path, rows)
        wall = [float(row["wall_latency_ms"]) for row in rows]
        gpu = [float(row["gpu_latency_ms"]) for row in rows]
        summary.update({
            "status": "complete",
            "completed_utc": now(),
            "raw_csv": str(raw_path),
            "raw_csv_sha256": sha256_file(raw_path),
            "mean_wall_latency_ms": statistics.mean(wall),
            "median_wall_latency_ms": statistics.median(wall),
            "p95_wall_latency_ms": percentile(wall, 0.95),
            "mean_gpu_latency_ms": statistics.mean(gpu),
            "median_gpu_latency_ms": statistics.median(gpu),
            "p95_gpu_latency_ms": percentile(gpu, 0.95),
            "decisions_per_second": 1000.0 / statistics.mean(wall),
            "peak_vram_mib": torch.cuda.max_memory_allocated(device) / (1024 * 1024),
        })
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        message = str(exc)
        if isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in message.lower():
            torch.cuda.empty_cache()
            summary.update({"status": "oom", "completed_utc": now(), "error": message})
        else:
            summary.update({"status": "failed", "completed_utc": now(), "error": message})
            atomic_json(args.output, summary)
            raise
    atomic_json(args.output, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
