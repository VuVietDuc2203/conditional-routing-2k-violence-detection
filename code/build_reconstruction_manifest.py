#!/usr/bin/env python3
"""Build a path-sanitized retained-cohort reconstruction manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


FIELDS = [
    "video_id",
    "source_dataset",
    "label",
    "split",
    "semantic_group_id",
    "source_video_relative",
    "original_dataset_relative",
    "merged_filename",
    "clip_length",
    "frame_count",
    "sample_fps",
    "m1_cached_tensor_sha256",
]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sanitize_original(value: str) -> str:
    normalized = value.replace("\\", "/")
    marker = "/datasets/raw/"
    position = normalized.lower().find(marker)
    if position < 0:
        raise ValueError(f"cannot sanitize original source path: {value}")
    return normalized[position + len(marker) :]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lineage", type=Path, required=True)
    parser.add_argument("--m1-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()

    lineage_rows = read_csv(args.lineage)
    m1_rows = read_csv(args.m1_manifest)
    retained = {row["video_id"]: row for row in lineage_rows if row["status"] == "included"}
    m1 = {row["video_id"]: row for row in m1_rows}
    if len(retained) != 3516 or len(m1) != 3516 or set(retained) != set(m1):
        raise ValueError("retained lineage and M1 manifest must contain the same 3516 unique video IDs")

    output_rows: list[dict[str, str]] = []
    for video_id in sorted(retained):
        left, right = retained[video_id], m1[video_id]
        if left["source_dataset"] != right["source_dataset"] or left["label"] != right["label"] or left["new_split"] != right["split"]:
            raise ValueError(f"lineage/M1 metadata mismatch: {video_id}")
        output_rows.append(
            {
                "video_id": video_id,
                "source_dataset": left["source_dataset"],
                "label": left["label"],
                "split": left["new_split"],
                "semantic_group_id": left["semantic_group_id"],
                "source_video_relative": right["source_video"].replace("\\", "/"),
                "original_dataset_relative": sanitize_original(right["original_path"]),
                "merged_filename": right["merged_filename"],
                "clip_length": right["clip_length"],
                "frame_count": right["frame_count"],
                "sample_fps": right["sample_fps"],
                "m1_cached_tensor_sha256": right["sha256"],
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)
    metadata = {
        "schema_version": "retained_source_reconstruction_v10_v1",
        "rows": len(output_rows),
        "splits": {name: sum(row["split"] == name for row in output_rows) for name in ("train", "val", "test")},
        "lineage_sha256": digest(args.lineage),
        "m1_manifest_sha256": digest(args.m1_manifest),
        "output_sha256": digest(args.output),
        "path_sanitization": "absolute prefix through datasets/raw removed; separators normalized to slash",
        "sha256_field_boundary": "m1_cached_tensor_sha256 hashes the cached whole-frame T50 tensor, not the third-party raw video",
        "raw_video_sha256_available": False,
    }
    args.metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
