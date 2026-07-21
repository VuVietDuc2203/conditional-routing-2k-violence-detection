#!/usr/bin/env python3
"""Remove host-specific paths from frozen replay selection sidecars.

The output preserves selection order, test IDs, labels, encoded timing, and
content hashes. It drops only `original_path` and `resolved_source`, which are
absolute paths on the evaluation host and are not required for reconstruction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


WORKLOADS = ("normal_only", "mixed_controlled", "kinetic_rich")
DROP_FIELDS = {"original_path", "resolved_source"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean(row: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in row.items() if key not in DROP_FIELDS}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--builder-manifest", type=Path, required=True)
    args = parser.parse_args()
    with args.canonical_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        canonical_rows = list(csv.DictReader(handle))
    canonical = {str(row["video_id"]): row for row in canonical_rows}
    if len(canonical) != len(canonical_rows):
        raise ValueError("corpus lineage contains duplicate video IDs")
    with args.builder_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        builder_rows = list(csv.DictReader(handle))
    builder = {str(row["video_id"]): row for row in builder_rows}
    if len(builder) != len(builder_rows):
        raise ValueError("workload-builder manifest contains duplicate video IDs")
    builder_test = {key: row for key, row in builder.items() if str(row["split"]) == "test"}
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "schema_version": "jrtip_workload_sidecars_v10_v1",
        "sanitization": "original_path and resolved_source removed; all other fields preserved",
        "corpus_lineage_sha256": sha256(args.canonical_manifest),
        "builder_input_manifest_sha256": sha256(args.builder_manifest),
        "builder_test_pool_n": len(builder_test),
        "workloads": {},
        "files": {},
    }
    for workload in WORKLOADS:
        source_dir = args.input_root / workload
        target_dir = args.output_root / workload
        target_dir.mkdir(parents=True, exist_ok=True)
        selection_path = source_dir / "selection.csv"
        with selection_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = [clean(dict(row)) for row in csv.DictReader(handle)]
        if not rows:
            raise ValueError(f"empty workload selection: {workload}")
        if any(str(row.get("runtime_label_access")) != "forbidden" for row in rows):
            raise ValueError(f"runtime label boundary mismatch: {workload}")
        for row in rows:
            video_id = str(row["video_id"])
            if video_id not in canonical:
                raise ValueError(f"workload ID is absent from corpus lineage: {video_id}")
            if int(row["label"]) != int(canonical[video_id]["label"]):
                raise ValueError(f"workload/canonical label mismatch: {video_id}")
            if video_id not in builder_test:
                raise ValueError(f"workload ID is absent from the builder's split=test pool: {video_id}")
            if int(row["label"]) != int(builder_test[video_id]["label"]):
                raise ValueError(f"workload/builder label mismatch: {video_id}")
        selection_out = target_dir / "selection.csv"
        with selection_out.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        workload_files = [selection_path]
        for name in ("sidecar_events.json", "sidecar_events_timeline.json"):
            path = source_dir / name
            if not path.exists():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, list):
                raise ValueError(f"unexpected sidecar payload: {path}")
            cleaned = [clean(dict(row)) for row in payload]
            (target_dir / name).write_text(json.dumps(cleaned, indent=2) + "\n", encoding="utf-8")
            workload_files.append(path)
        labels = Counter(int(row["label"]) for row in rows)
        sources = Counter(str(row["source_dataset"]) for row in rows)
        old_splits = Counter(str(canonical[str(row["video_id"])]["old_split"]) for row in rows)
        frozen_splits = Counter(str(canonical[str(row["video_id"])]["new_split"]) for row in rows)
        summary["workloads"][workload] = {
            "selected_segments": len(rows),
            "normal_segments": labels[0],
            "violence_segments": labels[1],
            "source_dataset_counts": dict(sorted(sources.items())),
            "builder_pool_split_counts": dict(sorted(old_splits.items())),
            "final_semantic_group_split_counts": dict(sorted(frozen_splits.items())),
            "selected_source_duration_sec": sum(float(row["duration_sec"]) for row in rows),
            "runtime_label_access": "forbidden",
        }
        for source_path in workload_files:
            relative = source_path.relative_to(args.input_root).as_posix()
            target_path = args.output_root / relative
            summary["files"][relative] = {
                "input_sha256": sha256(source_path),
                "sanitized_sha256": sha256(target_path),
            }
    manifest = args.input_root / "workload_manifest.json"
    if manifest.exists():
        target = args.output_root / "workload_build_manifest.json"
        target.write_text(manifest.read_text(encoding="utf-8"), encoding="utf-8")
        summary["files"]["workload_manifest.json"] = {
            "input_sha256": sha256(manifest),
            "sanitized_sha256": sha256(target),
        }
    builder_out = args.output_root / "builder_input_manifest.csv"
    cleaned_builder = [clean(dict(row)) for row in builder_rows]
    with builder_out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cleaned_builder[0]))
        writer.writeheader()
        writer.writerows(cleaned_builder)
    summary["files"]["builder_input_manifest.csv"] = {
        "input_sha256": sha256(args.builder_manifest),
        "sanitized_sha256": sha256(builder_out),
    }
    (args.output_root / "workload_composition_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
