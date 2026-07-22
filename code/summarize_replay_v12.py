#!/usr/bin/env python3
"""Verify v11 same-pipeline replay and derive paired/stage summaries for v12."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path


WORKLOADS = ("normal", "mixed", "kinetic")
MODES = ("m3_route_on", "m3_route_bypassed")
PAIR_METRICS = (
    "q_candidate", "q_update", "classifier_calls", "cpu_percent_mean",
    "achieved_analysis_fps", "latency_p95_ms", "deadline_miss_rate",
    "gpu_board_energy_j_per_update",
)
STAGE_FIELDS = ("total_ms", "yolo_ms", "yolo_inference_ms", "hdbscan_ms", "gate_ms", "crop_ms", "classifier_ms")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aggregate(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    marker_path = args.control_dir / "MATCHED_REPLAY_COMPLETE.json"
    summary_path = args.raw_dir / "matched_replay_summary.json"
    audit_path = args.raw_dir / "candidate_equivalence_audit.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8-sig"))
    payload = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    audit = json.loads(audit_path.read_text(encoding="utf-8-sig"))
    if marker.get("run_count") != 18 or payload.get("protocol") != "jvcir_same_pipeline_replay_v11":
        raise ValueError("unexpected replay completion marker or protocol")
    if sha256(summary_path) != marker["aggregate_sha256"]:
        raise ValueError("replay aggregate hash mismatch")
    if sha256(audit_path) != marker["candidate_equivalence_sha256"]:
        raise ValueError("candidate audit hash mismatch")
    if not audit.get("audit_pass") or len(audit.get("comparisons", [])) != 9:
        raise ValueError("candidate-equivalence audit did not pass 9/9 pairs")

    records = payload["records"]
    if len(records) != 18:
        raise ValueError(f"expected 18 process records, found {len(records)}")
    record_map = {(r["mode"], r["workload"], int(r["repeat"])): r for r in records}
    expected = {(m, w, r) for m in MODES for w in WORKLOADS for r in (1, 2, 3)}
    if set(record_map) != expected:
        raise ValueError("replay cells are incomplete or duplicated")

    paired_rows: list[dict[str, object]] = []
    for workload in WORKLOADS:
        for repeat in (1, 2, 3):
            route = record_map[("m3_route_on", workload, repeat)]
            bypass = record_map[("m3_route_bypassed", workload, repeat)]
            row: dict[str, object] = {"workload": workload, "repeat": repeat}
            for metric in PAIR_METRICS:
                route_value, bypass_value = float(route[metric]), float(bypass[metric])
                row[f"route_{metric}"] = route_value
                row[f"bypass_{metric}"] = bypass_value
                row[f"route_minus_bypass_{metric}"] = route_value - bypass_value
                row[f"route_relative_change_{metric}"] = route_value / bypass_value - 1.0 if bypass_value else None
            paired_rows.append(row)

    stage_runs: list[dict[str, object]] = []
    for key, record in sorted(record_map.items()):
        mode, workload, repeat = key
        trace_path = args.raw_dir / str(record["run_id"]) / "repeat_01" / "frame_trace.csv"
        with trace_path.open(encoding="utf-8-sig", newline="") as handle:
            trace = [
                row for row in csv.DictReader(handle)
                if row.get("analyzed", "").lower() == "true"
                and float(row.get("source_time_sec") or -1.0) >= 60.0
            ]
        if len(trace) != int(record["analyzed_frames"]) or len(trace) != 4800:
            raise ValueError(f"measured trace length mismatch for {record['run_id']}")
        stage_row: dict[str, object] = {"mode": mode, "workload": workload, "repeat": repeat, "n_updates": len(trace)}
        for field in STAGE_FIELDS:
            stage_row[f"{field}_mean"] = statistics.mean(float(row.get(field) or 0.0) for row in trace)
        stage_runs.append(stage_row)

    stage_summary: list[dict[str, object]] = []
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in stage_runs:
        grouped[(str(row["mode"]), str(row["workload"]))].append(row)
    for mode in MODES:
        for workload in WORKLOADS:
            item: dict[str, object] = {"mode": mode, "workload": workload}
            for field in STAGE_FIELDS:
                stats = aggregate([float(row[f"{field}_mean"]) for row in grouped[(mode, workload)]])
                for key, value in stats.items():
                    item[f"{field}_{key}"] = value
            stage_summary.append(item)

    comparison_summary: dict[str, object] = {}
    for workload in WORKLOADS:
        subset = [row for row in paired_rows if row["workload"] == workload]
        comparison_summary[workload] = {
            "candidate_call_reduction_fraction": aggregate([-float(row["route_relative_change_q_candidate"]) for row in subset]),
            "process_cpu_reduction_fraction": aggregate([-float(row["route_relative_change_cpu_percent_mean"]) for row in subset]),
            "board_energy_reduction_fraction": aggregate([-float(row["route_relative_change_gpu_board_energy_j_per_update"]) for row in subset]),
            "throughput_relative_change": aggregate([float(row["route_relative_change_achieved_analysis_fps"]) for row in subset]),
            "deadline_miss_percentage_point_change": aggregate([100.0 * float(row["route_minus_bypass_deadline_miss_rate"]) for row in subset]),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paired_path = args.output_dir / "replay_paired_deltas.csv"
    with paired_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)
    stage_runs_path = args.output_dir / "replay_stage_process_means.csv"
    with stage_runs_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(stage_runs[0]))
        writer.writeheader()
        writer.writerows(stage_runs)
    stage_summary_path = args.output_dir / "replay_stage_summary.csv"
    with stage_summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(stage_summary[0]))
        writer.writeheader()
        writer.writerows(stage_summary)
    result = {
        "protocol": "jvcir_v12_verified_same_pipeline_replay_postprocessing",
        "source_protocol": payload["protocol"],
        "aggregation": "process is the replicate; mean, sample SD, minimum and maximum across n=3 processes",
        "inference_boundary": "descriptive matched replay; no inferential p-values from n=3 process runs",
        "input_hashes": {
            "completion_marker_sha256": sha256(marker_path),
            "matched_replay_summary_sha256": sha256(summary_path),
            "candidate_equivalence_audit_sha256": sha256(audit_path),
        },
        "candidate_equivalence_pairs": len(audit["comparisons"]),
        "run_count": len(records),
        "aggregate_rows": payload["rows"],
        "paired_comparisons": comparison_summary,
        "stage_timing_note": "per-update stage means are descriptive and are not additive because detector inference is contained in the detector/tracker interval and an update may contain multiple classifier calls",
    }
    result_path = args.output_dir / "replay_v12_summary.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    completion = {
        "status": "complete",
        "protocol": result["protocol"],
        "run_count": len(records),
        "candidate_equivalence_pairs": len(audit["comparisons"]),
        "summary_sha256": sha256(result_path),
        "paired_deltas_sha256": sha256(paired_path),
        "stage_process_means_sha256": sha256(stage_runs_path),
        "stage_summary_sha256": sha256(stage_summary_path),
    }
    (args.output_dir / "REPLAY_V12_POSTPROCESS_COMPLETE.json").write_text(
        json.dumps(completion, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
