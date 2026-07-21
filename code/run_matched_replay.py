#!/usr/bin/env python3
"""Run and aggregate a paired continuous-reference versus routed replay campaign.

This wrapper does not change the streaming implementation. It calls the frozen
benchmark_streaming_2k_v10.py runner with identical source, timing and precision
arguments, while recording host metadata and process-level CPU/RAM samples.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil


WORKLOADS = {
    "normal": "normal_only_1440p30.mp4",
    "mixed": "mixed_controlled_1440p30.mp4",
    "kinetic": "kinetic_rich_1440p30.mp4",
}
MODES = ("m3_gated", "m1_dense_s1")
REPLICATES = (1, 2, 3)
IDLE_MEMORY_CEILING_MIB = 10_000


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def gpu_state() -> tuple[int, int]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    util, memory = result.stdout.strip().splitlines()[0].split(",")
    return int(util.strip()), int(memory.strip())


def wait_idle(path: Path, workload: str, mode: str, repeat: int) -> dict[str, list[int]]:
    samples: list[int] = []
    memories: list[int] = []
    for attempt in range(1, 241):
        attempt_utils: list[int] = []
        attempt_memory: list[int] = []
        for _ in range(4):
            util, memory = gpu_state()
            attempt_utils.append(util)
            attempt_memory.append(memory)
            time.sleep(2)
        passed = (
            statistics.median(attempt_utils) <= 8
            and max(attempt_utils) <= 12
            and max(attempt_memory) <= IDLE_MEMORY_CEILING_MIB
            and max(attempt_memory) - min(attempt_memory) <= 128
        )
        samples.extend(attempt_utils)
        memories.extend(attempt_memory)
        with path.open("a", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["utc", "workload", "mode", "repeat", "attempt", "sample", "utilization_percent", "memory_used_mib", "attempt_passed"],
            )
            if path.stat().st_size == 0:
                writer.writeheader()
            for index, (util, memory) in enumerate(zip(attempt_utils, attempt_memory), start=1):
                writer.writerow({"utc": now(), "workload": workload, "mode": mode, "repeat": repeat, "attempt": attempt, "sample": index, "utilization_percent": util, "memory_used_mib": memory, "attempt_passed": passed})
        if passed:
            return {"utilization_percent": attempt_utils, "memory_used_mib": attempt_memory}
        time.sleep(30)
    raise RuntimeError(f"strict idle gate timed out for {workload}/{mode}/r{repeat}")


def host_manifest(repo: Path, args: argparse.Namespace) -> dict[str, object]:
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    cpu_result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)"],
        capture_output=True,
        text=True,
        check=False,
    )
    cpu = cpu_result.stdout.strip() or platform.processor()
    vm = psutil.virtual_memory()
    return {
        "created_utc": now(),
        "hostname": platform.node(),
        "os": platform.platform(),
        "cpu_model": cpu,
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_processors": psutil.cpu_count(logical=True),
        "total_ram_bytes": vm.total,
        "total_ram_gib": vm.total / (1024 ** 3),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch_version": __import__("torch").__version__,
        "cuda_version": __import__("torch").version.cuda,
        "gpu_query": gpu,
        "benchmark_runner_sha256": sha256(repo / "validation_code" / "benchmark_streaming_2k_v10.py"),
        "wrapper_sha256": sha256(Path(__file__)),
        "duration_sec": args.duration_sec,
        "warmup_sec": args.warmup_sec,
        "analysis_fps": 8.0,
        "source_fps": 30.0,
        "loop_source": True,
        "source_loop_policy": "rewind the same hash-verified file only when needed to complete 60 s warm-up plus 600 s measured source time",
        "precision": "FP32 (--no-amp)",
        "label_access": "forbidden_in_runtime",
    }


def run_one(command: list[str], cwd: Path, stdout_path: Path, stderr_path: Path) -> dict[str, object]:
    started = time.perf_counter()
    samples: list[dict[str, float]] = []
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        proc = subprocess.Popen(command, cwd=cwd, stdout=stdout, stderr=stderr)
        child = psutil.Process(proc.pid)
        while proc.poll() is None:
            try:
                with child.oneshot():
                    samples.append({
                        "elapsed_sec": time.perf_counter() - started,
                        "cpu_percent": child.cpu_percent(interval=None),
                        "rss_bytes": child.memory_info().rss,
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            time.sleep(1)
        returncode = proc.wait()
    elapsed = time.perf_counter() - started
    return {
        "return_code": returncode,
        "wall_time_sec": elapsed,
        "cpu_percent_mean": statistics.mean([s["cpu_percent"] for s in samples]) if samples else None,
        "cpu_percent_peak": max([s["cpu_percent"] for s in samples], default=None),
        "rss_peak_bytes": max([s["rss_bytes"] for s in samples], default=None),
        "process_samples": samples,
    }


def load_summary(run_dir: Path) -> dict[str, object]:
    summary_path = run_dir / "repeat_01" / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    return json.loads(summary_path.read_text(encoding="utf-8"))


def aggregate(output_root: Path, records: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for record in records:
        grouped.setdefault((str(record["mode"]), str(record["workload"])), []).append(record)
    aggregate_rows: list[dict[str, object]] = []
    for (mode, workload), values in sorted(grouped.items()):
        if len(values) != 3:
            raise ValueError(f"expected 3 process runs for {mode}/{workload}, found {len(values)}")
        metric_names = [
            "achieved_analysis_fps", "classifier_calls", "q_update", "deadline_miss_rate", "deadline_miss_count",
            "latency_p50_ms", "latency_p95_ms", "latency_p99_ms", "gpu_util_mean_percent", "vram_peak_mb",
            "power_mean_w", "power_peak_w", "wall_time_sec", "cpu_percent_mean", "cpu_percent_peak", "rss_peak_bytes",
        ]
        row: dict[str, object] = {"mode": mode, "workload": workload, "n": 3}
        for metric in metric_names:
            nums = [float(v[metric]) for v in values if v.get(metric) is not None and v.get(metric) != "not_available"]
            if not nums:
                row[metric] = "not_available"
                continue
            mean = statistics.mean(nums)
            sd = statistics.stdev(nums) if len(nums) > 1 else 0.0
            row[metric] = mean
            row[f"{metric}_sample_sd"] = sd
            row[f"{metric}_min"] = min(nums)
            row[f"{metric}_max"] = max(nums)
        aggregate_rows.append(row)
    aggregate_path = output_root / "matched_replay_summary.json"
    payload = {
        "protocol": "matched_complete_replay_v10",
        "aggregation": "mean, sample SD, min and max across three independent process runs; process is replicate",
        "deadline_ms": 125.0,
        "rows": aggregate_rows,
        "records": records,
    }
    write_json(aggregate_path, payload)
    fields = sorted({key for row in aggregate_rows for key in row})
    with (output_root / "matched_replay_summary.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(aggregate_rows)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--m1-checkpoint", type=Path, required=True)
    parser.add_argument("--m3-checkpoint", type=Path, required=True)
    parser.add_argument("--duration-sec", type=float, default=600.0)
    parser.add_argument("--warmup-sec", type=float, default=60.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    output_root = args.output_root.resolve()
    analysis_dir = args.analysis_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    runner = repo / "validation_code" / "benchmark_streaming_2k_v10.py"
    if not runner.exists() or not args.m1_checkpoint.exists() or not args.m3_checkpoint.exists():
        raise FileNotFoundError("runner or checkpoint missing")
    m1_root = args.m1_checkpoint.parent.parent.parent
    m3_root = args.m3_checkpoint.parent.parent.parent
    host = host_manifest(repo, args)
    write_json(analysis_dir / "system_manifest.json", host)
    workload_root = repo / "result" / "streaming_2k" / "workloads_v1_10m"
    for source_name in WORKLOADS.values():
        if not (workload_root / source_name).exists():
            raise FileNotFoundError(workload_root / source_name)
    idle_path = analysis_dir / "idle_gate_samples.csv"
    log_dir = analysis_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    order = [("normal", "m3_gated"), ("normal", "m1_dense_s1"), ("mixed", "m1_dense_s1"), ("mixed", "m3_gated"), ("kinetic", "m3_gated"), ("kinetic", "m1_dense_s1")]
    for repeat in REPLICATES:
        for workload, mode in order:
            source = workload_root / WORKLOADS[workload]
            run_id = f"{mode}__{workload}__r{repeat}"
            run_root = output_root / run_id
            summary = run_root / "repeat_01" / "summary.json"
            if summary.exists() and args.resume:
                resource_path = run_root / "process_resource_samples.json"
                provenance_path = run_root / "run_provenance.json"
                if not resource_path.exists() or not provenance_path.exists():
                    raise FileNotFoundError(f"completed run lacks resource/provenance evidence: {run_id}")
                run_result = json.loads(resource_path.read_text(encoding="utf-8"))
            else:
                if run_root.exists():
                    raise FileExistsError(f"partial output exists; use --resume only for completed runs: {run_root}")
                idle_state = wait_idle(idle_path, workload, mode, repeat)
                checkpoint = args.m3_checkpoint if mode == "m3_gated" else args.m1_checkpoint
                command = [
                    str(args.python), str(runner), "--source", str(source), "--mode", mode,
                    "--output-dir", str(run_root), "--m1-root", str(m1_root), "--m3-root", str(m3_root),
                    "--threshold", "0.475", "--source-fps", "30", "--analysis-fps", "8",
                    "--width", "2560", "--height", "1440", "--duration-sec", str(args.duration_sec),
                    "--warmup-sec", str(args.warmup_sec), "--loop-source", "--repeat", "1", "--device", "cuda",
                    "--person-model", "yolo11n.pt", "--detector-device", "0", "--tracker", "bytetrack.yaml", "--no-amp",
                ]
                stdout_path = log_dir / f"{run_id}.stdout.log"
                stderr_path = log_dir / f"{run_id}.stderr.log"
                run_result = run_one(command, repo, stdout_path, stderr_path)
                if int(run_result["return_code"]) != 0:
                    raise RuntimeError(f"replay failed: {run_id}; see {stderr_path}")
                write_json(run_root / "process_resource_samples.json", run_result)
                write_json(run_root / "run_provenance.json", {
                    "run_id": run_id, "workload": workload, "mode": mode, "repeat": repeat,
                    "source_sha256": sha256(source), "checkpoint_sha256": sha256(checkpoint),
                    "idle_samples": idle_state, "command": command, "host_manifest_sha256": sha256(analysis_dir / "system_manifest.json"),
                })
            run_summary = load_summary(run_root)
            if str(run_summary.get("mode")) != mode or run_summary.get("label_access") != "forbidden_in_runtime":
                raise ValueError(f"protocol mismatch in {run_id}")
            if int(run_summary.get("width", 0)) != 2560 or int(run_summary.get("height", 0)) != 1440 or float(run_summary.get("analysis_fps", 0)) != 8.0:
                raise ValueError(f"resolution/analysis mismatch in {run_id}")
            if not bool(run_summary.get("loop_source")) or int(run_summary.get("source_loop_count", 0)) < 1:
                raise ValueError(f"source-loop protocol was not executed in {run_id}")
            measured_source_duration = float(run_summary.get("measured_source_duration_sec", 0.0))
            if measured_source_duration < float(args.duration_sec) - 0.25:
                raise ValueError(f"measured source time is incomplete in {run_id}: {measured_source_duration}")
            expected_updates = int(round(float(args.duration_sec) * 8.0))
            if abs(int(run_summary.get("analyzed_frames", 0)) - expected_updates) > 2:
                raise ValueError(f"analyzed-update denominator is incomplete in {run_id}")
            record = {
                "run_id": run_id, "workload": workload, "mode": mode, "repeat": repeat,
                "source_sha256": sha256(source), "checkpoint_sha256": str(run_summary.get("checkpoint_sha256")),
                **{key: run_summary.get(key) for key in ("achieved_analysis_fps", "classifier_calls", "analyzed_frames", "measured_source_duration_sec", "source_loop_count", "deadline_miss_rate", "deadline_miss_count", "latency_p50_ms", "latency_p95_ms", "latency_p99_ms", "gpu_util_mean_percent", "vram_peak_mb", "power_mean_w", "power_peak_w")},
                "q_update": float(run_summary.get("classifier_calls", 0)) / max(1, int(run_summary.get("analyzed_frames", 0))),
                "wall_time_sec": run_result.get("wall_time_sec"), "cpu_percent_mean": run_result.get("cpu_percent_mean"),
                "cpu_percent_peak": run_result.get("cpu_percent_peak"), "rss_peak_bytes": run_result.get("rss_peak_bytes"),
                "summary_sha256": sha256(run_root / "repeat_01" / "summary.json"),
            }
            records.append(record)
            write_json(analysis_dir / "matched_replay_run_ledger.json", records)
    payload = aggregate(output_root, records)
    marker = {
        "protocol": "matched_complete_replay_v10",
        "status": "complete",
        "created_utc": now(),
        "run_count": len(records),
        "system_manifest_sha256": sha256(analysis_dir / "system_manifest.json"),
        "aggregate_sha256": sha256(output_root / "matched_replay_summary.json"),
        "ledger_sha256": sha256(analysis_dir / "matched_replay_run_ledger.json"),
        "runner_sha256": sha256(Path(__file__)),
        "payload_rows": len(payload["rows"]),
    }
    write_json(analysis_dir / "MATCHED_REPLAY_COMPLETE.json", marker)
    print(json.dumps(marker, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
