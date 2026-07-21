#!/usr/bin/env python3
"""Freeze a portable one-clip/checkpoint bundle for two-GPU microbenchmarks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MODELS = ("c3d", "i3d", "resnet_lstm", "slowfast", "swin3d", "josenet")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--validation-freeze-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--units", type=int, default=100)
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    protocol = args.protocol_root.resolve()
    validation = args.validation_freeze_root.resolve()
    output = args.output_root.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Refusing non-empty bundle root: {output}")
    output.mkdir(parents=True, exist_ok=True)

    registry_path = protocol / "model_registry.yaml"
    registry_payload = json.loads(registry_path.read_text(encoding="utf-8"))
    registry = {row["model_id"]: row for row in registry_payload["models"]}
    inventory = {
        (row["model_id"], int(row["seed"])): row
        for row in read_csv(validation / "validation_run_inventory.csv")
    }
    test_ids_by_model: dict[str, set[str]] = {}
    manifests: dict[str, tuple[Path, list[dict[str, str]]]] = {}
    for model in MODELS:
        spec = registry[model]
        source = protocol / str(spec["full_manifest"])
        if sha256_file(source) != spec["full_manifest_sha256"]:
            raise RuntimeError(f"Manifest hash mismatch: {model}")
        rows = read_csv(source)
        test_rows = [row for row in rows if row["split"] == "test"]
        test_ids_by_model[model] = {row["video_id"] for row in test_rows}
        manifests[model] = (source, test_rows)
    common = set.intersection(*(test_ids_by_model[model] for model in MODELS))
    if len(common) != 526:
        raise RuntimeError(f"Expected 526 common test IDs, found {len(common)}")
    selected_id = sorted(common)[0]
    write_csv(
        output / "benchmark_ids.csv",
        [{"unit_index": index, "video_id": selected_id} for index in range(1, args.units + 1)],
        ["unit_index", "video_id"],
    )

    bundle_registry: list[dict[str, Any]] = []
    copied_cache: set[str] = set()
    for model in MODELS:
        spec = registry[model]
        row = next(item for item in manifests[model][1] if item["video_id"] == selected_id)
        manifest_target = output / "manifests" / f"{model}.csv"
        write_csv(manifest_target, [row], list(row))
        cache_relative = Path(row["cache_path"])
        cache_key = cache_relative.as_posix().lower()
        cache_source = repo / "result" / "gpu_cache" / cache_relative
        cache_target = output / "gpu_cache" / cache_relative
        if cache_key not in copied_cache:
            cache_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cache_source, cache_target)
            copied_cache.add(cache_key)

        inv = inventory[(model, 50900)]
        checkpoint_source = Path(inv["run_dir"]) / "best.pt"
        if sha256_file(checkpoint_source) != inv["checkpoint_sha256"]:
            raise RuntimeError(f"Checkpoint hash mismatch: {model}")
        checkpoint_target = output / "checkpoints" / model / "best.pt"
        checkpoint_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkpoint_source, checkpoint_target)
        bundle_registry.append({
            "model_id": model,
            "paper_name": spec["paper_name"],
            "clip_length": spec["clip_length"],
            "seed": 50900,
            "checkpoint": checkpoint_target.relative_to(output).as_posix(),
            "checkpoint_sha256": sha256_file(checkpoint_target),
            "manifest": manifest_target.relative_to(output).as_posix(),
            "manifest_sha256": sha256_file(manifest_target),
            "cache_profile": spec["cache_profile"],
        })
    atomic_json(output / "benchmark_registry.json", {"models": bundle_registry})

    files = sorted(path for path in output.rglob("*") if path.is_file())
    manifest_rows = [{
        "path": path.relative_to(output).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    } for path in files]
    write_csv(output / "BUNDLE_MANIFEST.csv", manifest_rows, ["path", "bytes", "sha256"])
    freeze = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_protocol": "one fixed common v2 test clip; batch-one model-core forward; 20 warmup; 3x100 measured units per GPU",
        "selected_video_id": selected_id,
        "unit_rows": args.units,
        "unique_video_ids": 1,
        "models": len(bundle_registry),
        "source_validation_freeze_sha256": sha256_file(validation / "VALIDATION_FREEZE.json"),
        "registry_sha256": sha256_file(output / "benchmark_registry.json"),
        "bundle_manifest_sha256": sha256_file(output / "BUNDLE_MANIFEST.csv"),
        "bundle_files_excluding_freeze": len(manifest_rows) + 1,
    }
    atomic_json(output / "BUNDLE_FREEZE.json", freeze)
    print(json.dumps(freeze, indent=2))


if __name__ == "__main__":
    main()
