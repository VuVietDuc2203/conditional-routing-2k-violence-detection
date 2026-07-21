#!/usr/bin/env python3
"""Validate the sealed v10 matched-replay evidence before manuscript use."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


EXPECTED_SOURCES = {
    "normal": "13408f70bb6988d345631eb011ec6b93448acd89c4097115f8f4014f4b1263fc",
    "mixed": "6cee6d9b78ce800049fb71a9cbaa428fc34bcc6d5e463bbd491ad2b325c79a8d",
    "kinetic": "557aa23bf1173bf512d0ecfb0b7844b3cf94e8ae5c8fd6500446740e5baee906",
}
EXPECTED_CHECKPOINTS = {
    "m3_gated": "004445a74504435a0e51fa4cb1c0c77659779ff3a1985877c6a08151ee0202ee",
    "m1_dense_s1": "547452c10cf354fc5578977163ba3c75d24f36243a6a3d10072930326b81111a",
}
EXPECTED_MODES = tuple(EXPECTED_CHECKPOINTS)
EXPECTED_WORKLOADS = tuple(EXPECTED_SOURCES)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    failures: list[str] = []
    checks: dict[str, object] = {}
    marker_path = args.analysis_dir / "MATCHED_REPLAY_COMPLETE.json"
    ledger_path = args.analysis_dir / "matched_replay_run_ledger.json"
    system_path = args.analysis_dir / "system_manifest.json"
    aggregate_path = args.raw_root / "matched_replay_summary.json"
    for path in (marker_path, ledger_path, system_path, aggregate_path):
        if not path.is_file():
            failures.append(f"missing required file: {path}")
    if failures:
        payload = {"status": "fail", "failures": failures, "checks": checks}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        raise SystemExit("; ".join(failures))

    marker = load(marker_path)
    ledger = load(ledger_path)
    system = load(system_path)
    aggregate = load(aggregate_path)
    assert isinstance(marker, dict) and isinstance(ledger, list)
    assert isinstance(system, dict) and isinstance(aggregate, dict)

    hash_targets = {
        "system_manifest_sha256": system_path,
        "aggregate_sha256": aggregate_path,
        "ledger_sha256": ledger_path,
    }
    for key, path in hash_targets.items():
        observed = sha256(path)
        if marker.get(key) != observed:
            failures.append(f"marker hash mismatch for {key}")
    checks["completion_marker_hashes"] = not any("marker hash" in item for item in failures)

    if marker.get("status") != "complete" or marker.get("protocol") != "matched_complete_replay_v10":
        failures.append("completion marker status/protocol mismatch")
    if aggregate.get("protocol") != "matched_complete_replay_v10":
        failures.append("aggregate protocol mismatch")
    if len(ledger) != 18 or marker.get("run_count") != 18:
        failures.append(f"expected 18 process records, found {len(ledger)}")

    expected_cells = {(mode, workload, repeat) for mode in EXPECTED_MODES for workload in EXPECTED_WORKLOADS for repeat in (1, 2, 3)}
    observed_cells = {(str(row.get("mode")), str(row.get("workload")), int(row.get("repeat", -1))) for row in ledger}
    if observed_cells != expected_cells:
        failures.append("mode/workload/repeat cells are incomplete or duplicated")
    if len({str(row.get("run_id")) for row in ledger}) != len(ledger):
        failures.append("run IDs are not unique")

    analyzed_by_workload: dict[str, set[int]] = defaultdict(set)
    source_hashes: dict[str, set[str]] = defaultdict(set)
    checkpoint_hashes: dict[str, set[str]] = defaultdict(set)
    q_values: dict[str, list[float]] = defaultdict(list)
    raw_hash_failures = 0
    resource_failures = 0
    for row in ledger:
        run_id = str(row["run_id"])
        mode = str(row["mode"])
        workload = str(row["workload"])
        run_root = args.raw_root / run_id
        summary_path = run_root / "repeat_01" / "summary.json"
        manifest_path = run_root / "repeat_01" / "run_manifest.json"
        provenance_path = run_root / "run_provenance.json"
        resource_path = run_root / "process_resource_samples.json"
        run_missing = False
        for path in (summary_path, manifest_path, provenance_path, resource_path):
            if not path.is_file():
                failures.append(f"missing raw run artifact: {path}")
                raw_hash_failures += 1
                run_missing = True
        if run_missing:
            continue
        summary = load(summary_path)
        manifest = load(manifest_path)
        provenance = load(provenance_path)
        resources = load(resource_path)
        assert isinstance(summary, dict) and isinstance(manifest, dict)
        assert isinstance(provenance, dict) and isinstance(resources, dict)
        if sha256(summary_path) != row.get("summary_sha256"):
            failures.append(f"summary hash mismatch: {run_id}")
            raw_hash_failures += 1
        source_hashes[workload].add(str(row.get("source_sha256")))
        checkpoint_hashes[mode].add(str(row.get("checkpoint_sha256")))
        if provenance.get("source_sha256") != row.get("source_sha256") or provenance.get("checkpoint_sha256") != row.get("checkpoint_sha256"):
            failures.append(f"ledger/provenance hash mismatch: {run_id}")
        if summary.get("label_access") != "forbidden_in_runtime" or manifest.get("label_access") != "forbidden_in_runtime":
            failures.append(f"runtime label boundary mismatch: {run_id}")
        if int(summary.get("width", 0)) != 2560 or int(summary.get("height", 0)) != 1440 or float(summary.get("analysis_fps", 0)) != 8.0:
            failures.append(f"spatial/sampling protocol mismatch: {run_id}")
        if not bool(summary.get("loop_source")) or int(summary.get("source_loop_count", 0)) != 1:
            failures.append(f"source-loop protocol mismatch: {run_id}")
        if float(summary.get("measured_source_duration_sec", 0.0)) < 599.75:
            failures.append(f"measured source interval is shorter than 600 s: {run_id}")
        analyzed = int(row.get("analyzed_frames", 0))
        if abs(analyzed - 4800) > 2:
            failures.append(f"expected approximately 4,800 measured updates: {run_id} has {analyzed}")
        analyzed_by_workload[workload].add(analyzed)
        q_value = float(row.get("q_update", -1))
        q_values[mode].append(q_value)
        if float(row.get("wall_time_sec", 0) or 0) <= 0 or float(row.get("rss_peak_bytes", 0) or 0) <= 0:
            failures.append(f"missing process timing/RSS: {run_id}")
            resource_failures += 1
        samples = resources.get("process_samples", [])
        if not isinstance(samples, list) or len(samples) < 10:
            failures.append(f"insufficient process resource samples: {run_id}")
            resource_failures += 1

    for workload, expected in EXPECTED_SOURCES.items():
        if source_hashes[workload] != {expected}:
            failures.append(f"source hash mismatch for {workload}: {sorted(source_hashes[workload])}")
        if len(analyzed_by_workload[workload]) != 1:
            failures.append(f"analyzed-update denominator differs within {workload}: {sorted(analyzed_by_workload[workload])}")
    for mode, expected in EXPECTED_CHECKPOINTS.items():
        if checkpoint_hashes[mode] != {expected}:
            failures.append(f"checkpoint hash mismatch for {mode}: {sorted(checkpoint_hashes[mode])}")
    if any(abs(value - 1.0) > 1e-12 for value in q_values["m1_dense_s1"]):
        failures.append("continuous reference did not call MoViNet exactly once per analyzed update")
    if any(value < 0.0 for value in q_values["m3_gated"]):
        failures.append("routed q_update is negative")

    counts = Counter((str(row.get("mode")), str(row.get("workload"))) for row in ledger)
    if set(counts.values()) != {3} or len(counts) != 6:
        failures.append(f"replicate counts are invalid: {dict(counts)}")
    aggregate_rows = aggregate.get("rows", [])
    if not isinstance(aggregate_rows, list) or len(aggregate_rows) != 6 or any(int(row.get("n", 0)) != 3 for row in aggregate_rows):
        failures.append("aggregate does not contain six n=3 rows")

    system_expected = {
        "duration_sec": 600.0,
        "warmup_sec": 60.0,
        "analysis_fps": 8.0,
        "source_fps": 30.0,
        "precision": "FP32 (--no-amp)",
        "label_access": "forbidden_in_runtime",
        "loop_source": True,
    }
    for key, expected in system_expected.items():
        if system.get(key) != expected:
            failures.append(f"system manifest mismatch for {key}: {system.get(key)!r}")

    checks.update(
        {
            "run_count": len(ledger),
            "cell_counts": {f"{mode}/{workload}": count for (mode, workload), count in sorted(counts.items())},
            "source_hashes": {key: sorted(value) for key, value in source_hashes.items()},
            "checkpoint_hashes": {key: sorted(value) for key, value in checkpoint_hashes.items()},
            "analyzed_frames_by_workload": {key: sorted(value) for key, value in analyzed_by_workload.items()},
            "raw_hash_failures": raw_hash_failures,
            "resource_failures": resource_failures,
            "process_is_replicate": aggregate.get("aggregation"),
            "dense_q_update_exactly_one": bool(q_values["m1_dense_s1"]) and all(abs(value - 1.0) <= 1e-12 for value in q_values["m1_dense_s1"]),
        }
    )
    output = {"status": "pass" if not failures else "fail", "failures": failures, "checks": checks}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if failures:
        raise SystemExit("matched replay validation failed; see output JSON")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
