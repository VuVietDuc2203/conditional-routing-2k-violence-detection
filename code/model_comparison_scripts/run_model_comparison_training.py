#!/usr/bin/env python3
"""Resume-safe serial smoke/training orchestrator for the frozen model matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_sha256(command: list[str]) -> str:
    return hashlib.sha256(json.dumps(command, separators=(",", ":")).encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_registry(protocol_root: Path) -> list[dict[str, Any]]:
    return json.loads((protocol_root / "model_registry.yaml").read_text(encoding="utf-8"))["models"]


def expected_run_dir(output_root: Path, spec: dict[str, Any], seed: int) -> Path:
    base = output_root / f"seed_{seed}" / str(spec["model_id"])
    if spec["runner"] == "movinet":
        return base / f"variant_{spec['runner_variant']}" / f"t{spec['clip_length']}"
    return base / str(spec["runner_model"]) / f"t{spec['clip_length']}"


def terminal_artifacts(run_dir: Path) -> list[Path]:
    return [
        run_dir / "best.pt",
        run_dir / "validation_predictions.csv",
        run_dir / "validation_metrics.json",
        run_dir / "history.csv",
    ]


def build_command(
    python: str,
    repo: Path,
    protocol_root: Path,
    output_root: Path,
    spec: dict[str, Any],
    seed: int,
    smoke: bool,
    patience: int,
) -> list[str]:
    manifest = protocol_root / str(spec["development_manifest"])
    model_output = output_root / f"seed_{seed}" / str(spec["model_id"])
    common = [
        "--cache-root",
        str(repo / "result" / "gpu_cache"),
        "--manifest",
        str(manifest),
        "--output-root",
        str(model_output),
        "--clip-length",
        str(spec["clip_length"]),
        "--seed",
        str(seed),
        "--epochs",
        "30",
        "--patience",
        str(patience),
        "--batch-size",
        str(spec.get("batch_size", "auto")),
        "--num-workers",
        "0",
        "--throughput-samples",
        "16",
        "--development-only",
    ]
    if spec["runner"] == "movinet":
        command = [
            python,
            "-m",
            "training_code.run_movinet_cached_experiments",
            "--variant",
            str(spec["runner_variant"]),
            "--selection-metric",
            "balanced_composite",
            "--deterministic",
            *common,
        ]
    else:
        command = [
            python,
            "-m",
            "training_code.run_jrtip_cached_experiments",
            "--model",
            str(spec["runner_model"]),
            "--selection-metric",
            "balanced",
            "--amp",
            *common,
        ]
        if spec["pretraining"] != "scratch":
            command.append("--pretrained")
    if smoke:
        command.append("--smoke")
    return command


def write_ledger_csv(path: Path, records: list[dict[str, Any]]) -> None:
    columns = [
        "job_id",
        "model_id",
        "seed",
        "mode",
        "status",
        "started_utc",
        "finished_utc",
        "exit_code",
        "command_sha256",
        "run_dir",
        "best_sha256",
        "validation_predictions_sha256",
        "validation_metrics_sha256",
        "history_sha256",
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow({column: record.get(column) for column in columns})
    os.replace(temporary, path)


def artifacts_match(record: dict[str, Any], run_dir: Path) -> bool:
    files = terminal_artifacts(run_dir)
    fields = ["best_sha256", "validation_predictions_sha256", "validation_metrics_sha256", "history_sha256"]
    return record.get("status") == "complete" and all(
        path.exists() and sha256_file(path) == record.get(field) for path, field in zip(files, fields)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", choices=["smoke", "train"], required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--model-id", action="append", default=[])
    parser.add_argument("--seed", type=int, action="append", default=[])
    args = parser.parse_args()
    if args.patience <= 0:
        raise ValueError("--patience must be positive")

    repo = args.repo_root.resolve()
    protocol = args.protocol_root.resolve()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    freeze_hash = sha256_file(protocol / "protocol_freeze.yaml")
    registry_hash = sha256_file(protocol / "model_registry.yaml")
    registry = load_registry(protocol)
    if args.model_id:
        requested = set(args.model_id)
        registry = [spec for spec in registry if spec["model_id"] in requested]
        if {spec["model_id"] for spec in registry} != requested:
            raise ValueError("At least one requested model_id is absent from the frozen registry")
    seeds = args.seed or ([50900] if args.mode == "smoke" else [50900, 50901, 50902])
    if not set(seeds).issubset({50900, 50901, 50902}):
        raise ValueError("Seeds must be a subset of the frozen 50900/50901/50902 list")

    ledger_path = output / "training_ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else {
        "protocol_freeze_sha256": freeze_hash,
        "model_registry_sha256": registry_hash,
        "records": [],
    }
    if ledger["protocol_freeze_sha256"] != freeze_hash or ledger["model_registry_sha256"] != registry_hash:
        raise RuntimeError("Ledger protocol hashes do not match the frozen protocol")
    records: list[dict[str, Any]] = ledger["records"]

    for seed in seeds:
        for spec in registry:
            job_id = f"{args.mode}:{spec['model_id']}:seed_{seed}"
            command = build_command(
                args.python, repo, protocol, output, spec, seed, args.mode == "smoke", args.patience
            )
            cmd_hash = command_sha256(command)
            run_dir = expected_run_dir(output, spec, seed)
            prior = next((record for record in records if record["job_id"] == job_id), None)
            if prior is not None:
                if prior.get("command_sha256") != cmd_hash:
                    raise RuntimeError(f"Command drift for existing job {job_id}")
                if artifacts_match(prior, run_dir):
                    print(f"SKIP hash-valid completed job {job_id}", flush=True)
                    continue
                if prior.get("status") in {"running", "failed"}:
                    raise RuntimeError(f"Refusing automatic duplicate/retry for nonterminal job {job_id}")
            elif run_dir.exists() and any(run_dir.iterdir()):
                raise RuntimeError(f"Refusing unledgered non-empty run directory: {run_dir}")

            record = {
                "job_id": job_id,
                "model_id": spec["model_id"],
                "seed": seed,
                "mode": args.mode,
                "status": "running",
                "started_utc": now(),
                "finished_utc": None,
                "exit_code": None,
                "command": command,
                "command_sha256": cmd_hash,
                "run_dir": str(run_dir),
            }
            records.append(record)
            atomic_json(ledger_path, ledger)
            write_ledger_csv(output / "training_ledger.csv", records)
            log_path = output / "logs" / f"{spec['model_id']}_seed_{seed}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"START {job_id}", flush=True)
            with log_path.open("ab") as log_handle:
                process = subprocess.run(command, cwd=repo, stdout=log_handle, stderr=subprocess.STDOUT)
            record["finished_utc"] = now()
            record["exit_code"] = int(process.returncode)
            if process.returncode == 0 and all(path.exists() for path in terminal_artifacts(run_dir)):
                files = terminal_artifacts(run_dir)
                record.update(
                    {
                        "status": "complete",
                        "best_sha256": sha256_file(files[0]),
                        "validation_predictions_sha256": sha256_file(files[1]),
                        "validation_metrics_sha256": sha256_file(files[2]),
                        "history_sha256": sha256_file(files[3]),
                    }
                )
            else:
                record["status"] = "failed"
            atomic_json(ledger_path, ledger)
            write_ledger_csv(output / "training_ledger.csv", records)
            if record["status"] != "complete":
                raise RuntimeError(f"Job failed or omitted terminal artifacts: {job_id}; see {log_path}")
            print(f"COMPLETE {job_id}", flush=True)

    atomic_json(
        output / "TRAINING_COMPLETE.json",
        {
            "status": "complete",
            "mode": args.mode,
            "protocol_freeze_sha256": freeze_hash,
            "model_registry_sha256": registry_hash,
            "jobs": len(records),
            "ledger_sha256": sha256_file(ledger_path),
            "completed_utc": now(),
        },
    )


if __name__ == "__main__":
    main()
