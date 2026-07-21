#!/usr/bin/env python3
"""Evaluate all frozen baseline checkpoints once on the common 526-ID v2 test.

The attempt marker is created before constructing any test dataset.  The script
never trains, never changes validation-selected thresholds, and refuses a second
completed attempt.
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
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


MODELS = ("c3d", "i3d", "resnet_lstm", "slowfast", "swin3d", "josenet")
SEEDS = (50900, 50901, 50902)


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows([{column: row.get(column, "") for column in columns} for row in rows])
    os.replace(temporary, path)


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tn = fp = fn = tp = 0
    for row in rows:
        true = int(row["true_label"])
        pred = int(row["pred_label"])
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
    precision_1 = tp / (tp + fp) if tp + fp else 0.0
    f1_0 = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
    f1_1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {
        "n": total,
        "correct": tn + tp,
        "accuracy": (tn + tp) / total if total else 0.0,
        "balanced_accuracy": (recall_0 + recall_1) / 2,
        "precision": precision_1,
        "recall": recall_1,
        "f1": f1_1,
        "f1_macro": (f1_0 + f1_1) / 2,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def verify_validation_freeze(root: Path) -> tuple[dict[str, Any], str]:
    marker_path = root / "VALIDATION_FREEZE.json"
    freeze = load_json(marker_path)
    if freeze.get("status") != "complete" or freeze.get("test_accessed") is not False:
        raise RuntimeError("Validation freeze is not a completed test-blind freeze")
    for name, expected in freeze.get("outputs", {}).items():
        actual = sha256_file(root / name)
        if actual != expected:
            raise RuntimeError(f"Validation freeze output hash mismatch: {name}")
    if int(freeze.get("runs", 0)) != 18 or int(freeze.get("paired_rows", 0)) != 9540:
        raise RuntimeError("Validation freeze cardinality is not 18 runs / 9540 rows")
    return freeze, sha256_file(marker_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--validation-freeze-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    protocol = args.protocol_root.resolve()
    validation_root = args.validation_freeze_root.resolve()
    output = args.output_root.resolve()
    marker_path = output / "V2_TEST_ATTEMPT.json"
    if marker_path.exists():
        prior = load_json(marker_path)
        raise RuntimeError(
            f"A v2 attempt marker already exists (status={prior.get('status')}, "
            f"attempt_id={prior.get('attempt_id')}); refusing a second attempt"
        )
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Refusing non-empty unmarked test output root: {output}")
    output.mkdir(parents=True, exist_ok=True)

    freeze, freeze_sha = verify_validation_freeze(validation_root)
    registry_path = protocol / "model_registry.yaml"
    protocol_path = protocol / "protocol_freeze.yaml"
    if sha256_file(registry_path) != freeze["model_registry_sha256"]:
        raise RuntimeError("Registry hash differs from validation freeze")
    if sha256_file(protocol_path) != freeze["protocol_freeze_sha256"]:
        raise RuntimeError("Protocol hash differs from validation freeze")
    registry_payload = load_json(registry_path)
    registry = {row["model_id"]: row for row in registry_payload["models"]}
    inventory_rows = read_csv(validation_root / "validation_run_inventory.csv")
    threshold_rows = read_csv(validation_root / "validation_thresholds.csv")
    inventory = {(row["model_id"], int(row["seed"])): row for row in inventory_rows}
    thresholds = {(row["model_id"], int(row["seed"])): float(row["threshold"]) for row in threshold_rows}
    expected_keys = {(model, seed) for model in MODELS for seed in SEEDS}
    if set(inventory) != expected_keys or set(thresholds) != expected_keys:
        raise RuntimeError("Validation inventory/threshold matrix is not exactly 6 models x 3 seeds")

    attempt = {
        "status": "started",
        "attempt_id": str(uuid.uuid4()),
        "started_utc": now(),
        "completed_utc": None,
        "validation_freeze_sha256": freeze_sha,
        "protocol_freeze_sha256": freeze["protocol_freeze_sha256"],
        "model_registry_sha256": freeze["model_registry_sha256"],
        "planned_runs": 18,
        "test_ids_per_run": 526,
        "policy": "checkpoint-only inference; validation-frozen threshold; no training; no post-test tuning",
    }
    atomic_json(marker_path, attempt)  # must precede any test dataset construction
    ledger_path = output / "test_ledger.json"
    ledger: dict[str, Any] = {"attempt_id": attempt["attempt_id"], "records": []}
    atomic_json(ledger_path, ledger)

    sys.path.insert(0, str(repo))
    from data.processors.model_cache_adapters import build_cached_dataloader, make_model_cache_dataset
    from training_code.run_jrtip_cached_experiments import build_model, evaluate, set_runtime_optimizations

    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("The one-time evaluator requires CUDA")
    device = torch.device(args.device)
    all_predictions: list[dict[str, Any]] = []
    run_metrics: list[dict[str, Any]] = []
    source_metrics: list[dict[str, Any]] = []
    common_ids: set[str] | None = None
    common_labels: dict[str, int] | None = None

    for model_id in MODELS:
        spec = registry[model_id]
        full_manifest_path = protocol / str(spec["full_manifest"])
        if sha256_file(full_manifest_path) != spec["full_manifest_sha256"]:
            raise RuntimeError(f"Full manifest hash mismatch: {model_id}")
        full_manifest = read_csv(full_manifest_path)
        test_manifest = {row["video_id"]: row for row in full_manifest if row["split"] == "test"}
        if len(test_manifest) != 526:
            raise RuntimeError(f"{model_id} test manifest has {len(test_manifest)} IDs, expected 526")

        for seed in SEEDS:
            set_runtime_optimizations(seed)
            key = (model_id, seed)
            inv = inventory[key]
            threshold = thresholds[key]
            checkpoint_path = Path(inv["run_dir"]) / "best.pt"
            if sha256_file(checkpoint_path) != inv["checkpoint_sha256"]:
                raise RuntimeError(f"Frozen checkpoint hash mismatch: {model_id}/{seed}")
            job_id = f"test:{model_id}:seed_{seed}"
            record: dict[str, Any] = {
                "job_id": job_id,
                "model_id": model_id,
                "seed": seed,
                "status": "running",
                "started_utc": now(),
                "finished_utc": None,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": inv["checkpoint_sha256"],
                "threshold": threshold,
                "full_manifest_sha256": spec["full_manifest_sha256"],
            }
            ledger["records"].append(record)
            atomic_json(ledger_path, ledger)
            print(f"START {job_id}", flush=True)
            started = time.perf_counter()

            dataset = make_model_cache_dataset(
                model_id,
                "test",
                cache_root=repo / "result" / "gpu_cache",
                clip_length=int(spec["clip_length"]),
                normalize=True,
                manifest_path=full_manifest_path,
            )
            loader = build_cached_dataloader(
                dataset,
                batch_size=int(spec["batch_size"]),
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                drop_last=False,
            )
            model = build_model(model_id, pretrained=False, freeze_backbone=False, clip_length=int(spec["clip_length"])).to(device)
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if str(checkpoint.get("model_name")) != model_id or int(checkpoint.get("clip_length", -1)) != int(spec["clip_length"]):
                raise RuntimeError(f"Checkpoint identity mismatch: {model_id}/{seed}")
            model.load_state_dict(checkpoint["model"], strict=True)
            evaluated = evaluate(model, loader, device, use_amp=True, criterion=None)
            metadata = evaluated["metadata"]
            if len(metadata) != 526:
                raise RuntimeError(f"Missing test metadata: {model_id}/{seed} returned {len(metadata)} rows")
            predictions: list[dict[str, Any]] = []
            for label, score, meta in zip(evaluated["y_true"], evaluated["y_score"], metadata):
                video_id = str(meta.get("video_id", ""))
                if video_id not in test_manifest:
                    raise RuntimeError(f"Unexpected test ID in inference: {model_id}/{seed}/{video_id}")
                pred = int(float(score) >= threshold)
                manifest_row = test_manifest[video_id]
                if int(label) != int(manifest_row["label"]):
                    raise RuntimeError(f"Test label mismatch: {model_id}/{seed}/{video_id}")
                predictions.append({
                    "model_id": model_id,
                    "paper_name": spec["paper_name"],
                    "seed": seed,
                    "video_id": video_id,
                    "semantic_group_id": manifest_row["semantic_group_id"],
                    "source_dataset": manifest_row["source_dataset"],
                    "true_label": int(label),
                    "score_violence": float(score),
                    "threshold": threshold,
                    "pred_label": pred,
                    "correct": int(pred == int(label)),
                })
            ids = [row["video_id"] for row in predictions]
            labels = {row["video_id"]: int(row["true_label"]) for row in predictions}
            if len(ids) != 526 or len(set(ids)) != 526 or set(ids) != set(test_manifest):
                raise RuntimeError(f"Test cardinality/ID mismatch: {model_id}/{seed}")
            if common_ids is None:
                common_ids = set(ids)
                common_labels = labels
            elif set(ids) != common_ids or labels != common_labels:
                raise RuntimeError(f"Test cohort/labels differ across runs: {model_id}/{seed}")

            result_dir = output / f"seed_{seed}" / model_id
            prediction_path = result_dir / "predictions.csv"
            metric_path = result_dir / "metrics.json"
            write_csv(prediction_path, predictions, list(predictions[0]))
            computed = metrics(predictions)
            by_source: dict[str, list[dict[str, Any]]] = {}
            for row in predictions:
                by_source.setdefault(row["source_dataset"], []).append(row)
            source_block = {source: metrics(rows) for source, rows in sorted(by_source.items())}
            elapsed = time.perf_counter() - started
            payload = {
                "job_id": job_id,
                "paper_name": spec["paper_name"],
                "seed": seed,
                "threshold": threshold,
                "checkpoint_sha256": inv["checkpoint_sha256"],
                "manifest_sha256": spec["full_manifest_sha256"],
                "elapsed_seconds": elapsed,
                "metrics": computed,
                "source_metrics": source_block,
            }
            atomic_json(metric_path, payload)
            record.update({
                "status": "complete",
                "finished_utc": now(),
                "elapsed_seconds": elapsed,
                "predictions": str(prediction_path),
                "predictions_sha256": sha256_file(prediction_path),
                "metrics": str(metric_path),
                "metrics_sha256": sha256_file(metric_path),
                "correct": computed["correct"],
                "accuracy": computed["accuracy"],
                "f1_macro": computed["f1_macro"],
            })
            atomic_json(ledger_path, ledger)
            run_metrics.append({
                "model_id": model_id,
                "paper_name": spec["paper_name"],
                "seed": seed,
                "threshold": threshold,
                **computed,
                "elapsed_seconds": elapsed,
            })
            for source, values in source_block.items():
                source_metrics.append({"model_id": model_id, "seed": seed, "source_dataset": source, **values})
            all_predictions.extend(predictions)
            del model, checkpoint, dataset, loader, evaluated
            torch.cuda.empty_cache()
            print(f"COMPLETE {job_id}: {computed['correct']}/526", flush=True)

    if len(run_metrics) != 18 or len(all_predictions) != 18 * 526:
        raise RuntimeError("Final v2 cardinality mismatch")
    summary: list[dict[str, Any]] = []
    for model_id in MODELS:
        rows = [row for row in run_metrics if row["model_id"] == model_id]
        summary.append({
            "model_id": model_id,
            "paper_name": registry[model_id]["paper_name"],
            "seeds": 3,
            "mean_accuracy": statistics.mean(float(row["accuracy"]) for row in rows),
            "sd_accuracy": statistics.stdev(float(row["accuracy"]) for row in rows),
            "mean_balanced_accuracy": statistics.mean(float(row["balanced_accuracy"]) for row in rows),
            "mean_f1_macro": statistics.mean(float(row["f1_macro"]) for row in rows),
            "min_correct": min(int(row["correct"]) for row in rows),
            "max_correct": max(int(row["correct"]) for row in rows),
        })
    summary.sort(key=lambda row: (row["mean_accuracy"], row["mean_f1_macro"]), reverse=True)
    for rank, row in enumerate(summary, start=1):
        row["test_rank"] = rank

    write_csv(output / "test_seed_metrics.csv", run_metrics, list(run_metrics[0]))
    write_csv(output / "test_source_metrics.csv", source_metrics, list(source_metrics[0]))
    write_csv(output / "test_paired_predictions.csv", all_predictions, list(all_predictions[0]))
    write_csv(output / "test_summary.csv", summary, list(summary[0]))
    final_files = ["test_ledger.json", "test_seed_metrics.csv", "test_source_metrics.csv", "test_paired_predictions.csv", "test_summary.csv"]
    attempt.update({
        "status": "complete",
        "completed_utc": now(),
        "completed_runs": 18,
        "paired_rows": len(all_predictions),
        "common_test_ids": len(common_ids or set()),
        "outputs": {name: sha256_file(output / name) for name in final_files},
    })
    atomic_json(marker_path, attempt)
    print(json.dumps(attempt, indent=2))


if __name__ == "__main__":
    main()
