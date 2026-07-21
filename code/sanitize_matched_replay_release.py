#!/usr/bin/env python3
"""Create a release-safe copy of validated matched-replay evidence.

The internal evidence remains unchanged. This copy retains measured traces,
telemetry, resource samples, hashes, and aggregates while removing host names,
absolute executable/source/checkpoint paths, and the absolute command line.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path


ABSOLUTE_WINDOWS = re.compile(r"(?i)(?:[a-z]:\\|\\\\[a-z0-9_.-]+\\)")
ABSOLUTE_POSIX_USER = re.compile(r"/(?:home|users)/[^/\s]+/")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sanitized_summary(value: dict[str, object], run_id: str) -> dict[str, object]:
    result = dict(value)
    source = Path(str(result.get("source", "unknown"))).name
    result["source"] = f"workload_media/{source}"
    checkpoint = str(result.get("checkpoint", "not_applicable"))
    if checkpoint != "not_applicable":
        result["checkpoint"] = f"checkpoints/{result.get('mode', 'model')}_best.pt"
    result["release_run_id"] = run_id
    return result


def sanitized_provenance(value: dict[str, object]) -> dict[str, object]:
    result = dict(value)
    result.pop("command", None)
    result["command_release_note"] = (
        "The machine-specific absolute command line is omitted from the public copy. "
        "Executed options are preserved by the released wrapper, runner, system manifest, "
        "per-run summary, source/checkpoint hashes, and run identifier."
    )
    return result


def copy_json_sanitized(source: Path, target: Path, transform) -> None:
    value = load(source)
    if not isinstance(value, dict):
        raise TypeError(f"expected object: {source}")
    write_json(target, transform(value))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    validation = load(args.validation)
    if not isinstance(validation, dict) or validation.get("status") != "pass":
        raise ValueError("matched evidence must pass validation before release sanitization")
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite release directory: {args.output_root}")
    args.output_root.mkdir(parents=True)

    ledger_path = args.analysis_dir / "matched_replay_run_ledger.json"
    system_path = args.analysis_dir / "system_manifest.json"
    completion_path = args.analysis_dir / "MATCHED_REPLAY_COMPLETE.json"
    ledger = load(ledger_path)
    system = load(system_path)
    completion = load(completion_path)
    if not isinstance(ledger, list) or len(ledger) != 18:
        raise ValueError("expected 18 validated ledger records")
    if not isinstance(system, dict) or not isinstance(completion, dict):
        raise TypeError("invalid system/completion metadata")

    write_json(args.output_root / "matched_replay_run_ledger.json", ledger)
    shutil.copy2(args.raw_root / "matched_replay_summary.json", args.output_root / "matched_replay_summary.json")
    shutil.copy2(args.raw_root / "matched_replay_summary.csv", args.output_root / "matched_replay_summary.csv")
    shutil.copy2(args.validation, args.output_root / "matched_replay_validation.json")

    public_system = dict(system)
    public_system.pop("hostname", None)
    public_system.pop("python_executable", None)
    public_system["release_note"] = "Hostname and absolute Python executable path were removed from the public copy."
    write_json(args.output_root / "system_manifest.json", public_system)

    source_hashes: dict[str, str] = {}
    for record in ledger:
        if not isinstance(record, dict):
            raise TypeError("ledger record is not an object")
        run_id = str(record["run_id"])
        source_run = args.raw_root / run_id
        target_run = args.output_root / "runs" / run_id
        target_repeat = target_run / "repeat_01"
        target_repeat.mkdir(parents=True)
        for filename in ("frame_trace.csv", "telemetry.csv"):
            shutil.copy2(source_run / "repeat_01" / filename, target_repeat / filename)
        copy_json_sanitized(
            source_run / "repeat_01" / "summary.json",
            target_repeat / "summary.json",
            lambda value, rid=run_id: sanitized_summary(value, rid),
        )
        copy_json_sanitized(
            source_run / "repeat_01" / "run_manifest.json",
            target_repeat / "run_manifest.json",
            lambda value, rid=run_id: sanitized_summary(value, rid),
        )
        shutil.copy2(source_run / "process_resource_samples.json", target_run / "process_resource_samples.json")
        copy_json_sanitized(
            source_run / "run_provenance.json",
            target_run / "run_provenance.json",
            sanitized_provenance,
        )
        source_hashes[run_id] = sha256(source_run / "repeat_01" / "summary.json")

    manifest_files = []
    unsafe_hits = []
    for path in sorted(item for item in args.output_root.rglob("*") if item.is_file()):
        relative = path.relative_to(args.output_root).as_posix()
        if path.suffix.lower() in {".json", ".csv", ".txt", ".md", ".log"}:
            content = path.read_text(encoding="utf-8-sig", errors="replace")
            if ABSOLUTE_WINDOWS.search(content) or ABSOLUTE_POSIX_USER.search(content):
                unsafe_hits.append(relative)
        manifest_files.append({"path": relative, "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    if unsafe_hits:
        raise ValueError(f"absolute path remained in release copy: {unsafe_hits}")

    release = {
        "schema_version": "matched_replay_public_release_v10_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "validation_status": "pass",
        "run_count": len(ledger),
        "internal_completion_marker_sha256": sha256(completion_path),
        "internal_system_manifest_sha256": sha256(system_path),
        "internal_ledger_sha256": sha256(ledger_path),
        "internal_summary_sha256_by_run": source_hashes,
        "sanitization": "machine-specific hostname, executable/source/checkpoint paths, and absolute command line removed; numerical evidence retained",
        "files": manifest_files,
    }
    write_json(args.output_root / "MATCHED_REPLAY_RELEASE_COMPLETE.json", release)
    print(json.dumps({"status": "pass", "files": len(manifest_files), "runs": len(ledger)}, indent=2))


if __name__ == "__main__":
    main()

