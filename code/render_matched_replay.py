#!/usr/bin/env python3
"""Verify matched replay aggregates and render the v10 comparison figure/table."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


WORKLOADS = ["normal", "mixed", "kinetic"]
MODES = ["m3_gated", "m1_dense_s1"]
MODE_LABELS = {"m3_gated": "Conditional routing", "m1_dense_s1": "Continuous reference"}
COLORS = {"m3_gated": "#0072B2", "m1_dense_s1": "#D55E00"}
HATCHES = {"m3_gated": "", "m1_dense_s1": "///"}


def row_map(payload: dict) -> dict[tuple[str, str], dict]:
    rows = {(str(row["mode"]), str(row["workload"])): row for row in payload["rows"]}
    expected = {(mode, workload) for mode in MODES for workload in WORKLOADS}
    if set(rows) != expected:
        raise ValueError(f"expected six mode/workload rows, found {sorted(rows)}")
    for key, row in rows.items():
        if int(row["n"]) != 3:
            raise ValueError(f"matched replay must have n=3 process runs: {key}")
    return rows


def integrate_gpu_energy(raw_root: Path, records: list[dict]) -> dict[str, dict[str, float | int | bool]]:
    per_run: dict[str, dict[str, float | int | bool]] = {}
    for record in records:
        run_id = str(record["run_id"])
        telemetry_path = raw_root / run_id / "repeat_01" / "telemetry.csv"
        trace_path = raw_root / run_id / "repeat_01" / "frame_trace.csv"
        summary_path = raw_root / run_id / "repeat_01" / "summary.json"
        with telemetry_path.open("r", encoding="utf-8-sig", newline="") as fh:
            telemetry = list(csv.DictReader(fh))
        numeric: list[tuple[float, float]] = []
        for row in telemetry:
            try:
                numeric.append((float(row["elapsed_sec"]), float(row["power_w"])))
            except (TypeError, ValueError):
                continue
        if len(numeric) < 10:
            raise ValueError(f"insufficient numeric power samples: {run_id}")
        elapsed = [item[0] for item in numeric]
        gaps = [right - left for left, right in zip(elapsed, elapsed[1:])]
        if any(gap <= 0 for gap in gaps) or max(gaps) > 5.0:
            raise ValueError(f"invalid telemetry time base: {run_id}")
        energy_j = sum(
            0.5 * (numeric[index - 1][1] + numeric[index][1]) * gaps[index - 1]
            for index in range(1, len(numeric))
        )
        with trace_path.open("r", encoding="utf-8-sig", newline="") as fh:
            trace = list(csv.DictReader(fh))
        analyzed_all = sum(str(row.get("analyzed", "")).lower() == "true" for row in trace)
        if analyzed_all <= 0:
            raise ValueError(f"no analyzed updates in trace: {run_id}")
        measured = [
            row
            for row in trace
            if str(row.get("analyzed", "")).lower() == "true" and float(row.get("source_time_sec", -1)) >= 60.0
        ]
        if len(measured) != int(record["analyzed_frames"]):
            raise ValueError(f"trace/ledger analyzed-update mismatch: {run_id}")
        trace_calls = sum(int(row.get("classifier_calls", 0)) for row in measured)
        if trace_calls != int(record["classifier_calls"]):
            raise ValueError(f"trace/ledger classifier-call mismatch: {run_id}")
        stage_fields = {
            "stage_total_mean_ms": "total_ms",
            "stage_yolo_tracker_mean_ms": "yolo_ms",
            "stage_yolo_inference_mean_ms": "yolo_inference_ms",
            "stage_hdbscan_mean_ms": "hdbscan_ms",
            "stage_gate_mean_ms": "gate_ms",
        }
        stage_means = {
            output_key: statistics.mean(float(row.get(trace_key, 0) or 0) for row in measured)
            for output_key, trace_key in stage_fields.items()
        }
        duration = elapsed[-1] - elapsed[0]
        wall = float(record["wall_time_sec"])
        coverage = duration / wall
        if coverage < 0.90:
            raise ValueError(f"power telemetry covers only {coverage:.3f} of process time: {run_id}")
        run_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        per_run[run_id] = {
            "power_sample_count": len(numeric),
            "telemetry_duration_sec": duration,
            "telemetry_wall_coverage": coverage,
            "gpu_board_energy_j": energy_j,
            "analyzed_updates_including_warmup": analyzed_all,
            "gpu_board_energy_per_analyzed_update_j": energy_j / analyzed_all,
            "telemetry_valid": True,
            "measured_trace_updates": len(measured),
            "measured_trace_classifier_calls": trace_calls,
            "classifier_call_latency_p50_ms": float(run_summary["classifier_latency_p50_ms"]),
            "classifier_call_latency_p95_ms": float(run_summary["classifier_latency_p95_ms"]),
            **stage_means,
        }
    return per_run


def add_energy_aggregates(payload: dict, rows: dict[tuple[str, str], dict], per_run: dict[str, dict[str, float | int | bool]]) -> None:
    records_by_key: dict[tuple[str, str], list[dict]] = {}
    for record in payload["records"]:
        record.update(per_run[str(record["run_id"])])
        records_by_key.setdefault((str(record["mode"]), str(record["workload"])), []).append(record)
    for key, records in records_by_key.items():
        row = rows[key]
        for metric in (
            "gpu_board_energy_j",
            "gpu_board_energy_per_analyzed_update_j",
            "telemetry_wall_coverage",
            "stage_total_mean_ms",
            "stage_yolo_tracker_mean_ms",
            "stage_yolo_inference_mean_ms",
            "stage_hdbscan_mean_ms",
            "stage_gate_mean_ms",
            "classifier_call_latency_p50_ms",
            "classifier_call_latency_p95_ms",
        ):
            nums = [float(record[metric]) for record in records]
            row[metric] = statistics.mean(nums)
            row[f"{metric}_sample_sd"] = statistics.stdev(nums)
            row[f"{metric}_min"] = min(nums)
            row[f"{metric}_max"] = max(nums)


def derive_comparisons(rows: dict[tuple[str, str], dict]) -> dict[str, object]:
    comparisons: dict[str, object] = {
        "boundary": "Operational configuration comparison: media, host, precision, timing, and sampling are matched; each configuration retains its own frozen checkpoint and input profile.",
        "workloads": {},
    }
    for workload in WORKLOADS:
        routed = rows[("m3_gated", workload)]
        dense = rows[("m1_dense_s1", workload)]
        routed_energy = float(routed["gpu_board_energy_per_analyzed_update_j"])
        dense_energy = float(dense["gpu_board_energy_per_analyzed_update_j"])
        comparisons["workloads"][workload] = {
            "routed_over_continuous_throughput_ratio": float(routed["achieved_analysis_fps"]) / float(dense["achieved_analysis_fps"]),
            "routed_minus_continuous_deadline_miss_percentage_points": 100.0 * (float(routed["deadline_miss_rate"]) - float(dense["deadline_miss_rate"])),
            "routed_classifier_call_reduction_percent": 100.0 * (1.0 - float(routed["q_update"]) / float(dense["q_update"])),
            "routed_gpu_board_energy_reduction_percent": 100.0 * (1.0 - routed_energy / dense_energy),
        }
    return comparisons


def narrative_summary(rows: dict[tuple[str, str], dict], comparisons: dict[str, object]) -> dict[str, object]:
    """Return canonical headline ranges for manuscript consistency checks."""

    def metric_span(mode: str, metric: str, scale: float = 1.0) -> dict[str, float]:
        values = [scale * float(rows[(mode, workload)][metric]) for workload in WORKLOADS]
        return {"min": min(values), "max": max(values)}

    workload_comparisons = comparisons["workloads"]
    assert isinstance(workload_comparisons, dict)

    def comparison_span(metric: str) -> dict[str, float]:
        values = [float(workload_comparisons[workload][metric]) for workload in WORKLOADS]
        return {"min": min(values), "max": max(values)}

    summary: dict[str, object] = {
        "aggregation": "range of the six n=3 mode/workload aggregate rows; workload order is normal, mixed, kinetic-rich",
        "routed": {
            "analyzed_updates_per_sec": metric_span("m3_gated", "achieved_analysis_fps"),
            "calls_per_100_updates": metric_span("m3_gated", "q_update", 100.0),
            "deadline_miss_percent": metric_span("m3_gated", "deadline_miss_rate", 100.0),
            "gpu_board_energy_j_per_update": metric_span("m3_gated", "gpu_board_energy_per_analyzed_update_j"),
        },
        "continuous": {
            "analyzed_updates_per_sec": metric_span("m1_dense_s1", "achieved_analysis_fps"),
            "calls_per_100_updates": metric_span("m1_dense_s1", "q_update", 100.0),
            "deadline_miss_percent": metric_span("m1_dense_s1", "deadline_miss_rate", 100.0),
            "gpu_board_energy_j_per_update": metric_span("m1_dense_s1", "gpu_board_energy_per_analyzed_update_j"),
        },
        "routed_vs_continuous": {
            "throughput_ratio": comparison_span("routed_over_continuous_throughput_ratio"),
            "classifier_call_reduction_percent": comparison_span("routed_classifier_call_reduction_percent"),
            "gpu_board_energy_reduction_percent": comparison_span("routed_gpu_board_energy_reduction_percent"),
        },
    }
    token_specs = {
        "routed_throughput": (summary["routed"]["analyzed_updates_per_sec"], 2),
        "continuous_throughput": (summary["continuous"]["analyzed_updates_per_sec"], 2),
        "routed_calls_per_100": (summary["routed"]["calls_per_100_updates"], 2),
        "continuous_calls_per_100": (summary["continuous"]["calls_per_100_updates"], 2),
        "routed_deadline_miss_percent": (summary["routed"]["deadline_miss_percent"], 2),
        "continuous_deadline_miss_percent": (summary["continuous"]["deadline_miss_percent"], 2),
        "routed_board_energy_j_per_update": (summary["routed"]["gpu_board_energy_j_per_update"], 2),
        "continuous_board_energy_j_per_update": (summary["continuous"]["gpu_board_energy_j_per_update"], 2),
        "throughput_ratio": (summary["routed_vs_continuous"]["throughput_ratio"], 2),
        "classifier_call_reduction_percent": (summary["routed_vs_continuous"]["classifier_call_reduction_percent"], 2),
        "board_energy_reduction_percent": (summary["routed_vs_continuous"]["gpu_board_energy_reduction_percent"], 2),
    }
    tokens: dict[str, dict[str, str]] = {"en": {}, "vi": {}}
    for name, (span, digits) in token_specs.items():
        low = float(span["min"])
        high = float(span["max"])
        en = f"{low:.{digits}f}" if abs(high - low) < 0.5 * 10 ** (-digits) else f"{low:.{digits}f}--{high:.{digits}f}"
        tokens["en"][name] = en
        tokens["vi"][name] = en.replace("--", "–").replace(".", ",")
    summary["tokens"] = tokens
    return summary


def values(rows: dict[tuple[str, str], dict], mode: str, metric: str) -> tuple[list[float], list[float]]:
    means = [float(rows[(mode, workload)][metric]) for workload in WORKLOADS]
    sds = [float(rows[(mode, workload)][f"{metric}_sample_sd"]) for workload in WORKLOADS]
    return means, sds


def render_figure(rows: dict[tuple[str, str], dict], output: Path) -> None:
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8})
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.2), constrained_layout=True)
    x = np.arange(len(WORKLOADS))
    width = 0.36
    specifications = [
        ("achieved_analysis_fps", "Analyzed updates/s", "(a) End-to-end throughput", False),
        ("deadline_miss_rate", "Deadline misses (%)", "(b) Updates over 125 ms", True),
        ("q_update", "Calls per 100 updates", "(c) Temporal-classifier schedule", True),
    ]
    for axis, (metric, ylabel, title, percent) in zip(axes, specifications):
        for index, mode in enumerate(MODES):
            mean, sd = values(rows, mode, metric)
            scale = 100.0 if percent else 1.0
            axis.bar(
                x + (index - 0.5) * width,
                np.asarray(mean) * scale,
                width,
                yerr=np.asarray(sd) * scale,
                capsize=2.5,
                color=COLORS[mode],
                hatch=HATCHES[mode],
                edgecolor="black",
                linewidth=0.45,
                label=MODE_LABELS[mode],
            )
        axis.set_xticks(x, ["Normal", "Mixed", "Kinetic-rich"])
        axis.set_ylabel(ylabel)
        axis.set_title(title, loc="left", fontweight="bold")
        axis.grid(axis="y", color="#d9d9d9", linewidth=0.6)
        axis.set_axisbelow(True)
    axes[0].axhline(8.0, color="#333333", linestyle="--", linewidth=1.0, label="8-Hz target")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=3, frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def make_table(rows: dict[tuple[str, str], dict], output: Path) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\caption{Matched complete 2K replay on RTX~5090. Values are mean $\pm$ sample SD for $n=3$ process runs on identical media; each schedule retains its frozen checkpoint and input profile. A service-time miss exceeds 125~ms; no input queue is simulated. GPU energy covers the decode/analysis loop, including warm-up but excluding model initialization; it is not total-system energy.}",
        r"\label{tab:matched_replay}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}llrrrr@{}}",
        r"\toprule",
        r"Workload & Schedule & Analyzed updates/s & Calls/100 updates & Miss rate (\%) & GPU energy (J/update)\\",
        r"\midrule",
    ]
    for workload in WORKLOADS:
        label = {"normal": "Normal", "mixed": "Mixed", "kinetic": "Kinetic-rich"}[workload]
        for index, mode in enumerate(MODES):
            row = rows[(mode, workload)]
            prefix = label if index == 0 else ""
            schedule = MODE_LABELS[mode]
            fps = f"{float(row['achieved_analysis_fps']):.3f} $\\pm$ {float(row['achieved_analysis_fps_sample_sd']):.3f}"
            q = f"{100*float(row['q_update']):.2f} $\\pm$ {100*float(row['q_update_sample_sd']):.2f}"
            miss = f"{100*float(row['deadline_miss_rate']):.2f} $\\pm$ {100*float(row['deadline_miss_rate_sample_sd']):.2f}"
            energy = "not available"
            if row.get("gpu_board_energy_per_analyzed_update_j") != "not_available":
                energy = f"{float(row['gpu_board_energy_per_analyzed_update_j']):.2f} $\\pm$ {float(row['gpu_board_energy_per_analyzed_update_j_sample_sd']):.2f}"
            lines.append(f"{prefix} & {schedule} & {fps} & {q} & {miss} & {energy}\\\\")
        if workload != WORKLOADS[-1]:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def mean_sd(row: dict, metric: str, digits: int = 2, scale: float = 1.0) -> str:
    if row.get(metric) == "not_available" or f"{metric}_sample_sd" not in row:
        return "n/a"
    mean = scale * float(row[metric])
    sd = scale * float(row[f"{metric}_sample_sd"])
    return f"{mean:.{digits}f} $\\pm$ {sd:.{digits}f}"


def make_supplementary(rows: dict[tuple[str, str], dict], output: Path) -> None:
    lines = [
        r"\documentclass[9pt]{article}",
        r"\usepackage[landscape,a4paper,margin=13mm]{geometry}",
        r"\usepackage{booktabs}",
        r"\renewcommand{\thetable}{S\arabic{table}}",
        r"\setcounter{table}{2}",
        r"\title{Matched-Replay Supplement for\\\large Conditional Routing for 2K Violence Detection: Accuracy, Invocation Cost, and Cross-Device Evaluation}",
        r"\author{Duc Viet Vu}",
        r"\date{}",
        r"\begin{document}",
        r"\maketitle",
        r"\vspace{-8mm}",
        r"\begin{table}[ht]",
        r"\caption{Process and device telemetry for the matched replay. Values are mean $\pm$ sample SD across three independent processes. Process wall time includes initialization and the 60-s source-time warm-up. Analysis-call latency covers RGB conversion and runtime processing in the 600-s measured interval but excludes source decoding and queueing. CPU follows \texttt{psutil} process semantics and may exceed 100\%; RSS is process peak resident memory. GPU values are NVIDIA device telemetry.}",
        r"\centering\scriptsize",
        r"\begin{tabular}{@{}llrrrrrrrrr@{}}",
        r"\toprule",
        r"Workload & Schedule & Wall (s) & Lat. p50 & Lat. p95 & Lat. p99 & CPU (\%) & RSS (GiB) & GPU (\%) & VRAM (MiB) & Power (W)\\",
        r"\midrule",
    ]
    for workload in WORKLOADS:
        label = {"normal": "Normal", "mixed": "Mixed", "kinetic": "Kinetic-rich"}[workload]
        for index, mode in enumerate(MODES):
            row = rows[(mode, workload)]
            lines.append(
                " & ".join(
                    [
                        label if index == 0 else "",
                        MODE_LABELS[mode],
                        mean_sd(row, "wall_time_sec", 1),
                        mean_sd(row, "latency_p50_ms", 1),
                        mean_sd(row, "latency_p95_ms", 1),
                        mean_sd(row, "latency_p99_ms", 1),
                        mean_sd(row, "cpu_percent_mean", 1),
                        mean_sd(row, "rss_peak_bytes", 2, 1.0 / (1024**3)),
                        mean_sd(row, "gpu_util_mean_percent", 1),
                        mean_sd(row, "vram_peak_mb", 0),
                        mean_sd(row, "power_mean_w", 1),
                    ]
                )
                + r"\\"
            )
        if workload != WORKLOADS[-1]:
            lines.append(r"\addlinespace")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        r"\begin{table}[ht]",
        r"\caption{Per-update stage timing from the measured replay traces (ms; mean $\pm$ sample SD across process means). YOLO/tracker is the complete detector--tracker call, while YOLO inference is the detector's reported inference subset and must not be added to it. Classifier latency is summarized over all MoViNet calls in the same process, including source-time warm-up; the runner did not retain every per-call latency separately in the measured interval. Routed updates may emit more than one call; therefore the columns are not an additive latency decomposition.}",
        r"\centering\scriptsize",
        r"\begin{tabular}{@{}llrrrrrrr@{}}",
        r"\toprule",
        r"Workload & Schedule & Total & YOLO/tracker & YOLO infer. & HDBSCAN & Gate & Classifier p50 & Classifier p95\\",
        r"\midrule",
    ]
    for workload in WORKLOADS:
        label = {"normal": "Normal", "mixed": "Mixed", "kinetic": "Kinetic-rich"}[workload]
        for index, mode in enumerate(MODES):
            row = rows[(mode, workload)]
            lines.append(
                " & ".join(
                    [
                        label if index == 0 else "",
                        MODE_LABELS[mode],
                        mean_sd(row, "stage_total_mean_ms", 2),
                        mean_sd(row, "stage_yolo_tracker_mean_ms", 2),
                        mean_sd(row, "stage_yolo_inference_mean_ms", 2),
                        mean_sd(row, "stage_hdbscan_mean_ms", 2),
                        mean_sd(row, "stage_gate_mean_ms", 2),
                        mean_sd(row, "classifier_call_latency_p50_ms", 2),
                        mean_sd(row, "classifier_call_latency_p95_ms", 2),
                    ]
                )
                + r"\\"
            )
        if workload != WORKLOADS[-1]:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", r"\end{document}"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--supplementary", type=Path, required=True)
    parser.add_argument("--verified-csv", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.summary.read_text(encoding="utf-8"))
    if payload.get("protocol") != "matched_complete_replay_v10":
        raise ValueError("unexpected matched replay protocol")
    rows = row_map(payload)
    per_run_energy = integrate_gpu_energy(args.raw_root, payload["records"])
    add_energy_aggregates(payload, rows, per_run_energy)
    payload["comparisons"] = derive_comparisons(rows)
    payload["narrative_summary"] = narrative_summary(rows, payload["comparisons"])
    verified_summary = args.verified_csv.with_suffix(".json")
    verified_summary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    render_figure(rows, args.figure)
    make_table(rows, args.table)
    make_supplementary(rows, args.supplementary)
    fields = sorted({key for row in rows.values() for key in row})
    with args.verified_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows([rows[(mode, workload)] for workload in WORKLOADS for mode in MODES])


if __name__ == "__main__":
    main()
