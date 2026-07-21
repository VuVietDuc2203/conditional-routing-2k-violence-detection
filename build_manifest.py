#!/usr/bin/env python3
"""Build a deterministic SHA-256 manifest for the public artifact directory."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    excluded = {"ARTIFACT_MANIFEST.json", "ARTIFACT_SHA256.txt"}
    files = [
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and path.name not in excluded
        and "__pycache__" not in path.parts
        and path.suffix.lower() != ".pyc"
    ]
    rows = [
        {"path": path.relative_to(ROOT).as_posix(), "size_bytes": path.stat().st_size, "sha256": digest(path)}
        for path in sorted(files, key=lambda item: item.relative_to(ROOT).as_posix())
    ]
    payload = {
        "schema_version": "jrtip_artifact_manifest_v10_v1",
        "file_count": len(rows),
        "files": rows,
        "public_release_status": "pending GitHub/Zenodo deposit",
        "raw_third_party_video_included": False,
    }
    (ROOT / "ARTIFACT_MANIFEST.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (ROOT / "ARTIFACT_SHA256.txt").write_text(
        "".join(f"{row['sha256']}  {row['path']}\n" for row in rows), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
