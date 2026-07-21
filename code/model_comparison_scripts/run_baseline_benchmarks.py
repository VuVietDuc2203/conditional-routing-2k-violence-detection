#!/usr/bin/env python3
"""Resume-safe serial runner for 6 models x 3 benchmark processes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MODELS = ("c3d", "i3d", "resnet_lstm", "slowfast", "swin3d", "josenet")


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


def gpu_sample() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    line = subprocess.check_output(command, text=True).strip().splitlines()[0]
    name, utilization, used, total = [item.strip() for item in line.split(",")]
    return {
        "gpu_name": name,
        "utilization_percent": int(utilization),
        "memory_used_mib": int(used),
        "memory_total_mib": int(total),
        "sampled_utc": now(),
    }


def idle_gate(samples: list[dict[str, Any]], max_median: float, max_peak: int) -> dict[str, Any]:
    utilization = [int(row["utilization_percent"]) for row in samples]
    memory = [int(row["memory_used_mib"]) for row in samples]
    result = {
        "samples": samples,
        "median_utilization_percent": statistics.median(utilization),
        "peak_utilization_percent": max(utilization),
        "memory_spread_mib": max(memory) - min(memory),
    }
    result["pass"] = bool(result["median_utilization_percent"] <= max_median and result["peak_utilization_percent"] <= max_peak)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--unit-script", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device-tag", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--units", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--idle-max-median", type=float, default=12.0)
    parser.add_argument("--idle-max-peak", type=int, default=20)
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    bundle = args.bundle_root.resolve()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    registry_path = bundle / "benchmark_registry.json"
    freeze_path = bundle / "BUNDLE_FREEZE.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "complete" or sha256_file(registry_path) != freeze.get("registry_sha256"):
        raise RuntimeError("Benchmark bundle freeze is invalid")
    registry_payload = json.loads(registry_path.read_text(encoding="utf-8"))
    registry = {row["model_id"]: row for row in registry_payload["models"]}
    if set(registry) != set(MODELS):
        raise RuntimeError("Benchmark registry model set mismatch")

    ledger_path = output / "benchmark_ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else {
        "device_tag": args.device_tag,
        "bundle_freeze_sha256": sha256_file(freeze_path),
        "records": [],
    }
    if ledger["bundle_freeze_sha256"] != sha256_file(freeze_path):
        raise RuntimeError("Benchmark bundle drift for existing ledger")

    for model_id in MODELS:
        spec = registry[model_id]
        for repeat in range(1, args.repeats + 1):
            job_id = f"{args.device_tag}:{model_id}:r{repeat}"
            result_path = output / model_id / f"r{repeat}.json"
            prior = next((row for row in ledger["records"] if row["job_id"] == job_id), None)
            if prior is not None:
                if prior.get("status") in {"complete", "oom"} and result_path.exists() and sha256_file(result_path) == prior.get("result_sha256"):
                    print(f"SKIP hash-valid {job_id}", flush=True)
                    continue
                raise RuntimeError(f"Refusing automatic retry of nonterminal/invalid benchmark: {job_id}")

            samples = []
            for _ in range(4):
                samples.append(gpu_sample())
                time.sleep(1.0)
            gate = idle_gate(samples, args.idle_max_median, args.idle_max_peak)
            if not gate["pass"]:
                raise RuntimeError(f"Idle GPU gate failed before {job_id}: {gate}")
            command = [
                sys.executable,
                str(args.unit_script.resolve()),
                "--repo-root", str(repo),
                "--cache-root", str(bundle / "gpu_cache"),
                "--model-id", model_id,
                "--clip-length", str(spec["clip_length"]),
                "--checkpoint", str(bundle / spec["checkpoint"]),
                "--checkpoint-sha256", spec["checkpoint_sha256"],
                "--manifest", str(bundle / spec["manifest"]),
                "--manifest-sha256", spec["manifest_sha256"],
                "--benchmark-ids", str(bundle / "benchmark_ids.csv"),
                "--output", str(result_path),
                "--repeat-index", str(repeat),
                "--warmup", str(args.warmup),
                "--units", str(args.units),
            ]
            record = {
                "job_id": job_id,
                "model_id": model_id,
                "repeat_index": repeat,
                "status": "running",
                "started_utc": now(),
                "finished_utc": None,
                "idle_gate": gate,
                "command": command,
            }
            ledger["records"].append(record)
            atomic_json(ledger_path, ledger)
            log_path = output / "logs" / f"{model_id}_r{repeat}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"START {job_id}", flush=True)
            with log_path.open("w", encoding="utf-8") as log_handle:
                process = subprocess.run(command, cwd=repo, stdout=log_handle, stderr=subprocess.STDOUT)
            if process.returncode != 0 or not result_path.exists():
                record.update({"status": "failed", "finished_utc": now(), "exit_code": process.returncode})
                atomic_json(ledger_path, ledger)
                raise RuntimeError(f"Benchmark failed: {job_id}; see {log_path}")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("status") not in {"complete", "oom"}:
                raise RuntimeError(f"Invalid benchmark terminal status for {job_id}: {result.get('status')}")
            record.update({
                "status": result["status"],
                "finished_utc": now(),
                "exit_code": process.returncode,
                "result": str(result_path),
                "result_sha256": sha256_file(result_path),
                "mean_wall_latency_ms": result.get("mean_wall_latency_ms"),
                "decisions_per_second": result.get("decisions_per_second"),
                "peak_vram_mib": result.get("peak_vram_mib"),
            })
            atomic_json(ledger_path, ledger)
            print(f"COMPLETE {job_id}: {result['status']}", flush=True)

    if len(ledger["records"]) != len(MODELS) * args.repeats:
        raise RuntimeError("Benchmark ledger cardinality mismatch")
    marker = {
        "status": "complete",
        "completed_utc": now(),
        "device_tag": args.device_tag,
        "runs": len(ledger["records"]),
        "complete_runs": sum(row["status"] == "complete" for row in ledger["records"]),
        "oom_runs": sum(row["status"] == "oom" for row in ledger["records"]),
        "ledger_sha256": sha256_file(ledger_path),
        "bundle_freeze_sha256": sha256_file(freeze_path),
    }
    atomic_json(output / "BENCHMARK_COMPLETE.json", marker)
    print(json.dumps(marker, indent=2))


if __name__ == "__main__":
    main()
