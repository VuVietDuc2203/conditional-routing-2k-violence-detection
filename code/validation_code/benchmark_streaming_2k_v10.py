"""Reproducible one-camera 1440p30 / 8-Hz-analytics replay benchmark.

Ground-truth labels intentionally are not accepted by this program.  Attach
labels only in a separate offline evaluator after a run has been sealed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training_code.m3_streaming_runtime import M3StreamingRuntime, RuntimeConfig, state_machine_self_test


TRACE_FIELDS = [
    "source_frame", "source_time_sec", "source_loop", "analyzed", "warmed_up", "total_ms", "deadline_miss",
    "gate_activations", "classifier_calls", "prediction", "score", "yolo_ms", "yolo_inference_ms", "hdbscan_ms",
    "gate_ms", "crop_ms", "classifier_ms",
]
TELEMETRY_FIELDS = ["elapsed_sec", "gpu_util_percent", "memory_used_mb", "power_w", "temperature_c"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], point: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * point / 100.0
    lower, upper = math.floor(index), math.ceil(index)
    if lower == upper:
        return float(ordered[lower])
    return float(ordered[lower] * (upper - index) + ordered[upper] * (index - lower))


def numeric_telemetry(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]


def gpu_telemetry() -> dict[str, float | str]:
    command = [
        "nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        line = subprocess.check_output(command, text=True, timeout=5).strip().splitlines()[0]
        def parse(item: str) -> float | str:
            value = item.strip()
            return float(value) if value not in {"[N/A]", "N/A"} else "not_available"
        util, memory, power, temp = [parse(item) for item in line.split(",")]
        return {"gpu_util_percent": util, "memory_used_mb": memory, "power_w": power, "temperature_c": temp}
    except Exception:
        return {key: "not_available" for key in TELEMETRY_FIELDS[1:]}


def parse_source(raw: str) -> int | str:
    return int(raw) if raw.isdecimal() else raw


def make_capture(source: int | str, width: int, height: int, fps: float) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW if isinstance(source, int) else cv2.CAP_ANY)
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")
    return cap


def checkpoint_for(args: argparse.Namespace) -> Path | None:
    if args.mode == "m3_gate_only":
        return None
    root = args.m3_root if args.mode == "m3_gated" else args.m1_root
    variant = "M3" if args.mode == "m3_gated" else "M1"
    path = root / f"variant_{variant}" / "t50" / "best.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    return path


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_once(args: argparse.Namespace, repeat: int) -> dict[str, Any]:
    output = args.output_dir / f"repeat_{repeat:02d}"
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = checkpoint_for(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    cfg = RuntimeConfig(threshold=args.threshold)
    runtime = M3StreamingRuntime(
        cfg, device, args.mode, checkpoint, args.person_model, args.detector_device,
        args.tracker, args.amp,
    )
    parsed_source = parse_source(args.source)
    cap = make_capture(parsed_source, args.width, args.height, args.source_fps)
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or args.source_fps)
    source_frame_count_reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    trace: list[dict[str, Any]] = []
    telemetry: list[dict[str, Any]] = []
    classifier_latencies: list[float] = []
    total_latencies: list[float] = []
    frame_index, analyzed, deadline_misses, calls, activations = 0, 0, 0, 0, 0
    source_loop_count = 0
    next_analysis_at, next_telemetry_at = 0.0, 0.0
    started = time.perf_counter()
    measurement_started: float | None = None
    try:
        while True:
            ok, bgr = cap.read()
            if not ok or bgr is None:
                target_source_time = args.warmup_sec + args.duration_sec if args.duration_sec else None
                can_loop = args.loop_source and not isinstance(parsed_source, int)
                if can_loop and target_source_time is not None and frame_index / source_fps < target_source_time:
                    if not cap.set(cv2.CAP_PROP_POS_FRAMES, 0):
                        raise RuntimeError("Failed to rewind replay source before requested measured duration")
                    ok, bgr = cap.read()
                    if not ok or bgr is None:
                        raise RuntimeError("Replay source remained unreadable after rewind")
                    source_loop_count += 1
                else:
                    break
            source_time = frame_index / source_fps
            elapsed = time.perf_counter() - started
            # Offline replay must consume a fixed amount of *source video*.
            # Wall-clock stopping biases slow methods by silently truncating
            # their workload, exactly what this benchmark is intended to show.
            if args.duration_sec and source_time >= args.duration_sec + args.warmup_sec:
                break
            decoded_height, decoded_width = bgr.shape[:2]
            if (decoded_width, decoded_height) != (args.width, args.height):
                raise RuntimeError(
                    f"Expected {args.width}x{args.height}; source delivered {decoded_width}x{decoded_height}"
                )
            if source_time >= args.warmup_sec and measurement_started is None:
                measurement_started = time.perf_counter()
            record: dict[str, Any] = {
                "source_frame": frame_index, "source_time_sec": source_time, "source_loop": source_loop_count, "analyzed": False,
                "warmed_up": False, "total_ms": 0.0, "deadline_miss": False,
                "gate_activations": 0, "classifier_calls": 0,
                "prediction": "", "score": "", "yolo_ms": 0.0, "yolo_inference_ms": 0.0,
                "hdbscan_ms": 0.0, "gate_ms": 0.0, "crop_ms": 0.0, "classifier_ms": 0.0,
            }
            if source_time + 1e-9 >= next_analysis_at:
                next_analysis_at += 1.0 / args.analysis_fps
                analyze_start = time.perf_counter()
                result = runtime.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                record["analyzed"], record["warmed_up"] = True, result["warmed_up"]
                record["total_ms"] = (time.perf_counter() - analyze_start) * 1000.0
                record["deadline_miss"] = record["total_ms"] > (1000.0 / args.analysis_fps)
                deadline_misses += int(record["deadline_miss"])
                analyzed += 1
                total_latencies.append(float(record["total_ms"]))
                timing = result["timing"]
                record.update({key: float(timing.get(key, 0.0)) for key in
                               ("yolo_ms", "yolo_inference_ms", "hdbscan_ms", "gate_ms")})
                record["classifier_calls"] = len(result["calls"])
                calls += len(result["calls"])
                record["gate_activations"] = len(result.get("gate_activations", []))
                activations += int(record["gate_activations"])
                if result["calls"]:
                    last = result["calls"][-1]
                    record.update({key: last[key] for key in ("prediction", "score", "crop_ms", "classifier_ms")})
                    classifier_latencies.extend(float(call["classifier_ms"]) for call in result["calls"])
            if elapsed >= next_telemetry_at:
                next_telemetry_at += 1.0
                telemetry.append({"elapsed_sec": elapsed, **gpu_telemetry()})
            trace.append(record)
            frame_index += 1
    finally:
        cap.release()
    measured = [row for row in trace if row["analyzed"] and row["source_time_sec"] >= args.warmup_sec]
    measured_latencies = [float(row["total_ms"]) for row in measured]
    measured_source_duration = (
        min(
            float(args.duration_sec),
            float(measured[-1]["source_time_sec"]) - float(measured[0]["source_time_sec"]) + 1.0 / args.analysis_fps,
        )
        if measured and args.duration_sec
        else 0.0
    )
    duration = max(1e-9, time.perf_counter() - (measurement_started or started))
    manifest = {
        "protocol": "one_camera_1440p30_ingest_8fps_analytics_t50",
        "mode": args.mode, "source": str(args.source), "width": args.width, "height": args.height,
        "source_fps_requested": args.source_fps, "source_fps_reported": source_fps,
        "source_frame_count_reported": source_frame_count_reported,
        "loop_source": bool(args.loop_source), "source_loop_count": source_loop_count,
        "analysis_fps": args.analysis_fps, "clip_length": cfg.clip_length,
        "thresholds": vars(cfg),
        "checkpoint": str(checkpoint) if checkpoint is not None else "not_loaded",
        "checkpoint_sha256": sha256_file(checkpoint) if checkpoint is not None else "not_applicable",
        "device": str(device), "torch": torch.__version__, "cuda": torch.version.cuda,
        "python": sys.version, "platform": platform.platform(), "label_access": "forbidden_in_runtime",
    }
    summary = {
        **manifest,
        "source_frames": frame_index, "analyzed_frames": len(measured),
        "measured_source_duration_sec": measured_source_duration,
        "achieved_analysis_fps": len(measured) / duration,
        "gate_activations": sum(row["gate_activations"] for row in measured),
        "gate_activations_per_min": sum(row["gate_activations"] for row in measured) * 60.0 / duration,
        "classifier_calls": sum(row["classifier_calls"] for row in measured),
        "classifier_calls_per_min": sum(row["classifier_calls"] for row in measured) * 60.0 / duration,
        "deadline_miss_count": sum(int(row["deadline_miss"]) for row in measured),
        "deadline_miss_rate": sum(int(row["deadline_miss"]) for row in measured) / max(1, len(measured)),
        "latency_p50_ms": percentile(measured_latencies, 50), "latency_p95_ms": percentile(measured_latencies, 95),
        "latency_p99_ms": percentile(measured_latencies, 99),
        "classifier_latency_p50_ms": percentile(classifier_latencies, 50),
        "classifier_latency_p95_ms": percentile(classifier_latencies, 95),
        "gpu_util_mean_percent": statistics.mean(numeric_telemetry(telemetry, "gpu_util_percent")) if numeric_telemetry(telemetry, "gpu_util_percent") else "not_available",
        "vram_peak_mb": max(numeric_telemetry(telemetry, "memory_used_mb"), default="not_available"),
        "power_mean_w": statistics.mean(numeric_telemetry(telemetry, "power_w")) if numeric_telemetry(telemetry, "power_w") else "not_available",
        "power_peak_w": max(numeric_telemetry(telemetry, "power_w"), default="not_available"),
    }
    write_csv(output / "frame_trace.csv", trace, TRACE_FIELDS)
    write_csv(output / "telemetry.csv", telemetry, TELEMETRY_FIELDS)
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Video file or local camera index, e.g. 0")
    parser.add_argument("--mode", choices=["m1_dense_s1", "m1_stride50", "m3_gated", "m3_gate_only"], required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("result/streaming_2k"))
    parser.add_argument("--m1-root", type=Path, default=Path("result/movinet_cached_experiments"))
    parser.add_argument("--m3-root", type=Path, default=Path("result/movinet_cppstack_t50_experiments"))
    parser.add_argument("--source-fps", type=float, default=30.0)
    parser.add_argument("--analysis-fps", type=float, default=8.0)
    parser.add_argument("--width", type=int, default=2560)
    parser.add_argument("--height", type=int, default=1440)
    parser.add_argument("--duration-sec", type=float, default=600.0)
    parser.add_argument("--warmup-sec", type=float, default=60.0)
    parser.add_argument("--loop-source", action="store_true", help="Rewind file input as needed so warm-up plus measured source time is complete")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--person-model", default="yolo11n.pt")
    parser.add_argument("--detector-device", default="0")
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        state_machine_self_test()
        print("state-machine self-test passed")
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summaries = [run_once(args, repeat) for repeat in range(1, args.repeat + 1)]
    (args.output_dir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
