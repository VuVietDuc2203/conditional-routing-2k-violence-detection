#!/usr/bin/env python3
"""Add raw-video SHA-256 values to the sanitized reconstruction manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    root = args.repo_root.resolve()
    input_sha256 = digest(args.input)
    with args.input.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("reconstruction manifest is empty")
    if len({row["video_id"] for row in rows}) != len(rows):
        raise ValueError("video_id is not unique")

    missing: list[str] = []
    for row in rows:
        relative = Path(row["source_video_relative"])
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"path escapes repository root: {relative}") from exc
        if not candidate.is_file():
            missing.append(relative.as_posix())
            continue
        row["raw_video_size_bytes"] = str(candidate.stat().st_size)
        row["raw_video_sha256"] = digest(candidate)
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} retained videos; first: {missing[:5]}")

    fields = list(rows[0])
    for field in ("raw_video_size_bytes", "raw_video_sha256"):
        if field not in fields:
            fields.append(field)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    duplicate_hash_groups = Counter(row["raw_video_sha256"] for row in rows)
    payload = {
        "schema_version": "retained_raw_video_hashes_v10_v1",
        "rows": len(rows),
        "unique_video_ids": len({row["video_id"] for row in rows}),
        "unique_raw_video_sha256": len(duplicate_hash_groups),
        "hash_groups_with_more_than_one_retained_id": sum(count > 1 for count in duplicate_hash_groups.values()),
        "splits": dict(sorted(Counter(row["split"] for row in rows).items())),
        "source_datasets": dict(sorted(Counter(row["source_dataset"] for row in rows).items())),
        "input_sha256": input_sha256,
        "output_sha256": digest(args.output),
        "raw_video_sha256_available": True,
        "boundary": "Hashes authenticate locally retained third-party media bytes; the media are not redistributed.",
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
