#!/usr/bin/env python3
"""Independently audit and freeze the 18 baseline validation runs.

This script is deliberately test-blind: it reads only development manifests and
validation artifacts.  It selects a per-run decision threshold from validation
scores using the pre-registered grid and writes a hash-addressed freeze consumed
by the one-time v2 evaluator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MODELS = ("c3d", "i3d", "resnet_lstm", "slowfast", "swin3d", "josenet")
SEEDS = (50900, 50901, 50902)
ARTIFACT_FIELDS = {
    "best.pt": "best_sha256",
    "validation_predictions.csv": "validation_predictions_sha256",
    "validation_metrics.json": "validation_metrics_sha256",
    "history.csv": "history_sha256",
}


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


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows([{column: row.get(column, "") for column in columns} for row in rows])
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def confusion(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    tn = fp = fn = tp = 0
    for row in rows:
        true = int(row["true_label"])
        pred = int(float(row["score_violence"]) >= threshold)
        if true == 0 and pred == 0:
            tn += 1
        elif true == 0 and pred == 1:
            fp += 1
        elif true == 1 and pred == 0:
            fn += 1
        else:
            tp += 1
    total = tn + fp + fn + tp
    recall_0 = tn / (tn + fp) if tn + fp else 0.0
    recall_1 = tp / (tp + fn) if tp + fn else 0.0
    f1_0 = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
    f1_1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {
        "n": total,
        "correct": tn + tp,
        "accuracy": (tn + tp) / total if total else 0.0,
        "balanced_accuracy": (recall_0 + recall_1) / 2,
        "f1_macro": (f1_0 + f1_1) / 2,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def select_threshold(rows: list[dict[str, Any]]) -> tuple[float, dict[str, Any]]:
    candidates: list[tuple[tuple[float, float, float, float], float, dict[str, Any]]] = []
    for step in range(61):
        threshold = round(0.35 + 0.005 * step, 3)
        metrics = confusion(rows, threshold)
        rank = (
            metrics["f1_macro"],
            metrics["balanced_accuracy"],
            -abs(threshold - 0.5),
            -threshold,
        )
        candidates.append((rank, threshold, metrics))
    _rank, threshold, metrics = max(candidates, key=lambda item: item[0])
    return threshold, metrics


def load_registry(path: Path) -> dict[str, dict[str, Any]]:
    payload = load_json(path)
    models = {str(row["model_id"]): row for row in payload["models"]}
    if set(models) != set(MODELS):
        raise RuntimeError(f"Registry model set mismatch: {sorted(models)}")
    return models


def ledger_records(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    if not isinstance(payload.get("records"), list):
        raise RuntimeError(f"Invalid ledger: {path}")
    return payload["records"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--c3d-root", type=Path, required=True)
    parser.add_argument("--seed50900-root", type=Path, required=True)
    parser.add_argument("--remaining-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    protocol = args.protocol_root.resolve()
    output = args.output_root.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty validation freeze: {output}")
    output.mkdir(parents=True, exist_ok=True)

    registry_path = protocol / "model_registry.yaml"
    protocol_path = protocol / "protocol_freeze.yaml"
    registry = load_registry(registry_path)
    protocol_freeze = load_json(protocol_path)
    if protocol_freeze.get("test_policy") is None:
        raise RuntimeError("Protocol lacks a test-access policy")

    c3d_ledger_path = args.c3d_root / "training_ledger.json"
    seed0_ledger_path = args.seed50900_root / "training_ledger.json"
    remaining_ledger_path = args.remaining_root / "training_ledger.json"
    ledgers = {
        "c3d": ledger_records(c3d_ledger_path),
        "seed0": ledger_records(seed0_ledger_path),
        "remaining": ledger_records(remaining_ledger_path),
    }

    selected: dict[tuple[str, int], tuple[dict[str, Any], Path, Path]] = {}
    for model in MODELS:
        for seed in SEEDS:
            if model == "c3d":
                source = "c3d"
                ledger_path = c3d_ledger_path
            elif seed == 50900:
                source = "seed0"
                ledger_path = seed0_ledger_path
            else:
                source = "remaining"
                ledger_path = remaining_ledger_path
            job_id = f"train:{model}:seed_{seed}"
            matches = [row for row in ledgers[source] if row.get("job_id") == job_id]
            if len(matches) != 1:
                raise RuntimeError(f"Expected one admissible ledger row for {job_id}; found {len(matches)}")
            record = matches[0]
            if record.get("status") != "complete" or int(record.get("exit_code", -1)) != 0:
                raise RuntimeError(f"Non-complete admissible job: {job_id}")
            selected[(model, seed)] = (record, Path(record["run_dir"]), ledger_path)

    inventory: list[dict[str, Any]] = []
    seed_metrics: list[dict[str, Any]] = []
    source_metrics: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    common_ids: set[str] | None = None
    common_labels: dict[str, int] | None = None

    for model in MODELS:
        spec = registry[model]
        manifest_path = protocol / str(spec["development_manifest"])
        if sha256_file(manifest_path) != str(spec["development_manifest_sha256"]):
            raise RuntimeError(f"Development manifest hash mismatch: {model}")
        manifest_rows = read_csv(manifest_path)
        val_manifest = {row["video_id"]: row for row in manifest_rows if row["split"] == "val"}
        if len(val_manifest) != 530:
            raise RuntimeError(f"{model} validation manifest has {len(val_manifest)} IDs, expected 530")

        for seed in SEEDS:
            record, run_dir, ledger_path = selected[(model, seed)]
            for filename, field in ARTIFACT_FIELDS.items():
                path = run_dir / filename
                actual = sha256_file(path)
                if actual != record.get(field):
                    raise RuntimeError(f"Artifact hash mismatch: {model}/{seed}/{filename}")

            metrics_path = run_dir / "validation_metrics.json"
            predictions_path = run_dir / "validation_predictions.csv"
            history_path = run_dir / "history.csv"
            stored = load_json(metrics_path)
            predictions = read_csv(predictions_path)
            history = read_csv(history_path)
            ids = [row["video_id"] for row in predictions]
            if len(predictions) != 530 or len(set(ids)) != 530:
                raise RuntimeError(f"{model}/{seed} predictions are not 530 unique rows")
            if set(ids) != set(val_manifest):
                raise RuntimeError(f"{model}/{seed} validation IDs differ from frozen manifest")
            labels = {row["video_id"]: int(row["true_label"]) for row in predictions}
            if any(labels[vid] != int(val_manifest[vid]["label"]) for vid in labels):
                raise RuntimeError(f"{model}/{seed} labels differ from frozen manifest")
            if common_ids is None:
                common_ids = set(ids)
                common_labels = labels
            elif set(ids) != common_ids or labels != common_labels:
                raise RuntimeError(f"{model}/{seed} does not share the common validation cohort/labels")
            if not history:
                raise RuntimeError(f"Empty training history: {model}/{seed}")

            raw_metrics = confusion(predictions, 0.5)
            for key in ("accuracy", "balanced_accuracy", "f1_macro"):
                if not math.isclose(float(stored[key]), float(raw_metrics[key]), rel_tol=0, abs_tol=1e-12):
                    raise RuntimeError(f"Stored metric mismatch for {model}/{seed}/{key}")
            threshold, selected_metrics = select_threshold(predictions)
            threshold_rows.append({
                "model_id": model,
                "paper_name": spec["paper_name"],
                "seed": seed,
                "threshold": threshold,
                **selected_metrics,
                "checkpoint_sha256": record["best_sha256"],
            })
            seed_metrics.append({
                "model_id": model,
                "paper_name": spec["paper_name"],
                "seed": seed,
                "threshold": threshold,
                **selected_metrics,
                "raw_argmax_accuracy": raw_metrics["accuracy"],
                "raw_argmax_balanced_accuracy": raw_metrics["balanced_accuracy"],
                "raw_argmax_f1_macro": raw_metrics["f1_macro"],
            })
            inventory.append({
                "model_id": model,
                "paper_name": spec["paper_name"],
                "seed": seed,
                "run_dir": str(run_dir),
                "ledger": str(ledger_path),
                "checkpoint_sha256": record["best_sha256"],
                "validation_predictions_sha256": record["validation_predictions_sha256"],
                "validation_metrics_sha256": record["validation_metrics_sha256"],
                "history_sha256": record["history_sha256"],
                "manifest_sha256": spec["development_manifest_sha256"],
                "actual_patience": 5,
                "validation_rows": 530,
                "history_epochs": len(history),
            })

            by_source: dict[str, list[dict[str, Any]]] = {}
            for row in predictions:
                source = val_manifest[row["video_id"]]["source_dataset"]
                by_source.setdefault(source, []).append(row)
                paired_rows.append({
                    "model_id": model,
                    "seed": seed,
                    "video_id": row["video_id"],
                    "semantic_group_id": val_manifest[row["video_id"]]["semantic_group_id"],
                    "source_dataset": source,
                    "true_label": int(row["true_label"]),
                    "score_violence": float(row["score_violence"]),
                    "pred_label": int(float(row["score_violence"]) >= threshold),
                    "correct": int(int(float(row["score_violence"]) >= threshold) == int(row["true_label"])),
                    "threshold": threshold,
                })
            for source, rows in sorted(by_source.items()):
                source_metrics.append({
                    "model_id": model,
                    "seed": seed,
                    "source_dataset": source,
                    **confusion(rows, threshold),
                })

    if len(inventory) != 18 or len(paired_rows) != 18 * 530:
        raise RuntimeError("Validation freeze row cardinality mismatch")

    summary: list[dict[str, Any]] = []
    for model in MODELS:
        rows = [row for row in seed_metrics if row["model_id"] == model]
        summary.append({
            "model_id": model,
            "paper_name": registry[model]["paper_name"],
            "seeds": 3,
            "mean_accuracy": statistics.mean(float(row["accuracy"]) for row in rows),
            "sd_accuracy": statistics.stdev(float(row["accuracy"]) for row in rows),
            "mean_balanced_accuracy": statistics.mean(float(row["balanced_accuracy"]) for row in rows),
            "sd_balanced_accuracy": statistics.stdev(float(row["balanced_accuracy"]) for row in rows),
            "mean_f1_macro": statistics.mean(float(row["f1_macro"]) for row in rows),
            "sd_f1_macro": statistics.stdev(float(row["f1_macro"]) for row in rows),
            "min_accuracy": min(float(row["accuracy"]) for row in rows),
            "max_accuracy": max(float(row["accuracy"]) for row in rows),
        })
    summary.sort(key=lambda row: (row["mean_accuracy"], row["mean_f1_macro"]), reverse=True)
    for rank, row in enumerate(summary, start=1):
        row["validation_rank"] = rank

    write_csv(output / "validation_run_inventory.csv", inventory, list(inventory[0]))
    write_csv(output / "validation_seed_metrics.csv", seed_metrics, list(seed_metrics[0]))
    write_csv(output / "validation_source_metrics.csv", source_metrics, list(source_metrics[0]))
    write_csv(output / "validation_paired_predictions.csv", paired_rows, list(paired_rows[0]))
    write_csv(output / "validation_thresholds.csv", threshold_rows, list(threshold_rows[0]))
    write_csv(output / "validation_summary.csv", summary, list(summary[0]))

    output_files = [
        "validation_run_inventory.csv",
        "validation_seed_metrics.csv",
        "validation_source_metrics.csv",
        "validation_paired_predictions.csv",
        "validation_thresholds.csv",
        "validation_summary.csv",
    ]
    freeze = {
        "status": "complete",
        "created_utc": now(),
        "test_accessed": False,
        "models": len(MODELS),
        "seeds_per_model": len(SEEDS),
        "runs": len(inventory),
        "validation_ids_per_run": 530,
        "paired_rows": len(paired_rows),
        "protocol_freeze_sha256": sha256_file(protocol_path),
        "model_registry_sha256": sha256_file(registry_path),
        "ledger_sha256": {
            "c3d_all_seeds": sha256_file(c3d_ledger_path),
            "seed50900_non_c3d_source": sha256_file(seed0_ledger_path),
            "remaining_seeds23": sha256_file(remaining_ledger_path),
        },
        "patience_note": "All 18 admitted commands used patience=5; protocol_v5 metadata retained an earlier patience=6 description and is not silently rewritten.",
        "outputs": {name: sha256_file(output / name) for name in output_files},
    }
    atomic_json(output / "VALIDATION_FREEZE.json", freeze)
    print(json.dumps(freeze, indent=2))


if __name__ == "__main__":
    main()
