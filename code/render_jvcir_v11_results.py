#!/usr/bin/env python3
"""Render manuscript/supplementary outputs from sealed v11 evidence only."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LABELS = {"normal": "Normal", "mixed": "Mixed", "kinetic": "Kinetic-rich"}
MODES = {"m3_route_on": "Route on", "m3_route_bypassed": "Route bypassed"}


def fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def mean_sd(row: dict[str, object], key: str, scale: float = 1.0, digits: int = 2) -> str:
    mean = float(row[key]) * scale
    sd = float(row.get(f"{key}_sample_sd", 0.0)) * scale
    return f"{mean:.{digits}f} $\\pm$ {sd:.{digits}f}"


def read_predictions(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def render_offline(summary: dict, predictions: list[dict[str, str]], manuscript: Path, supplementary: Path, figures: Path) -> None:
    route = summary["route_on"]
    bypass = summary["route_bypassed"]
    counts = summary["candidate_counts"]
    rates = summary["invocation_rates"]
    mcnemar = summary["paired_mcnemar"]
    boot = summary["grouped_bootstrap"]["accuracy_difference_route_on_minus_bypassed"]
    tex = rf"""At the final T50 decision update of each video, route bypassed classified all {counts['eligible']} eligible candidates and obtained {bypass['correct']}/526 ({100*bypass['accuracy']:.2f}\%) accuracy with macro-F1 {100*bypass['macro_f1']:.2f}\%. Route on selected {counts['route_selected']}/{counts['eligible']} candidates ($q_{{\mathrm{{candidate}}}}={100*float(rates['route_on_q_candidate']):.2f}\%$), called the classifier in {counts['route_selected_videos']}/526 videos ($q_{{\mathrm{{video}}}}={100*float(rates['route_on_q_video']):.2f}\%$), and obtained {route['correct']}/526 ({100*route['accuracy']:.2f}\%) accuracy with macro-F1 {100*route['macro_f1']:.2f}\%. Across all {counts['analyzed_updates']} sampled updates, $q_{{\mathrm{{update}}}}={float(rates['route_on_q_update']):.4f}$ for route on and {float(rates['route_bypassed_q_update']):.4f} for route bypassed. The paired accuracy difference (route on minus bypassed) was {100*(route['accuracy']-bypass['accuracy']):.2f} percentage points; the 95\% semantic-group bootstrap interval was [{100*boot['ci95_percentile'][0]:.2f}, {100*boot['ci95_percentile'][1]:.2f}] points. Exact McNemar discordance was {mcnemar['route_on_only_correct']} versus {mcnemar['route_bypassed_only_correct']} videos ($p={mcnemar['exact_two_sided_p']:.4f}$).

\begin{{table}}[t]
\caption{{Paired same-candidate policy ablation on 526 held-out videos. Both policies use identical eligible candidate tensors and M3 scores.}}
\label{{tab:same_candidate_offline}}
\centering
\small
\begin{{tabular}}{{@{{}}lrrrrr@{{}}}}
\toprule
Policy & Accuracy & Macro-F1 & TN & FP & FN/TP \\ \midrule
Route on & {100*route['accuracy']:.2f}\% & {100*route['macro_f1']:.2f}\% & {route['tn']} & {route['fp']} & {route['fn']}/{route['tp']} \\
Route bypassed & {100*bypass['accuracy']:.2f}\% & {100*bypass['macro_f1']:.2f}\% & {bypass['tn']} & {bypass['fp']} & {bypass['fn']}/{bypass['tp']} \\ \bottomrule
\end{{tabular}}
\end{{table}}
"""
    (manuscript / "same_pipeline_offline_results.tex").write_text(tex, encoding="utf-8")

    categories = {
        "Both correct": 0,
        "Route only correct": 0,
        "Bypass only correct": 0,
        "Both wrong": 0,
    }
    source_rows: dict[str, dict[str, int]] = {}
    for row in predictions:
        y = int(row["true_label"])
        rc = int(row["route_on_pred"]) == y
        bc = int(row["route_bypassed_pred"]) == y
        key = "Both correct" if rc and bc else "Route only correct" if rc else "Bypass only correct" if bc else "Both wrong"
        categories[key] += 1
        src = source_rows.setdefault(row["source_dataset"], {name: 0 for name in categories})
        src[key] += 1
    with (supplementary / "offline_policy_error_categories.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_dataset", *categories])
        writer.writeheader()
        writer.writerow({"source_dataset": "all", **categories})
        for source in sorted(source_rows):
            writer.writerow({"source_dataset": source, **source_rows[source]})

    colors = ["#2E7D32", "#1565C0", "#EF6C00", "#C62828"]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    names = list(categories)
    values = [categories[name] for name in names]
    bars = ax.bar(np.arange(len(names)), values, color=colors)
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_ylabel("Held-out videos")
    ax.set_xticks(np.arange(len(names)), ["Both\ncorrect", "Route only\ncorrect", "Bypass only\ncorrect", "Both\nwrong"])
    ax.set_title("Paired policy outcomes on identical candidate tensors")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figures / "Fig7.pdf", bbox_inches="tight")
    plt.close(fig)


def render_replay(summary: dict, candidate_audit: dict, manuscript: Path, supplementary: Path, figures: Path) -> None:
    rows = summary["rows"]
    lookup = {(row["mode"], row["workload"]): row for row in rows}
    table_lines = []
    for workload in ("normal", "mixed", "kinetic"):
        for mode in ("m3_route_on", "m3_route_bypassed"):
            row = lookup[(mode, workload)]
            table_lines.append(
                f"{LABELS[workload]} & {MODES[mode]} & {mean_sd(row, 'achieved_analysis_fps')} & "
                f"{mean_sd(row, 'q_candidate', 100)} & {mean_sd(row, 'q_update', 100)} & "
                f"{mean_sd(row, 'latency_p95_ms')} & {mean_sd(row, 'deadline_miss_rate', 100)} & "
                f"{mean_sd(row, 'gpu_board_energy_j_per_update')} \\\\"
            )
    route_rows = [lookup[("m3_route_on", w)] for w in ("normal", "mixed", "kinetic")]
    bypass_rows = [lookup[("m3_route_bypassed", w)] for w in ("normal", "mixed", "kinetic")]
    route_q = [100 * float(r["q_candidate"]) for r in route_rows]
    route_fps = [float(r["achieved_analysis_fps"]) for r in route_rows]
    bypass_fps = [float(r["achieved_analysis_fps"]) for r in bypass_rows]
    route_energy = [float(r["gpu_board_energy_j_per_update"]) for r in route_rows]
    bypass_energy = [float(r["gpu_board_energy_j_per_update"]) for r in bypass_rows]
    tex = rf"""The candidate-equivalence audit passed for all {len(candidate_audit['comparisons'])} policy--workload--replicate pairs. Route on classified {min(route_q):.2f}--{max(route_q):.2f}\% of eligible candidates. Its workload-mean throughput was {min(route_fps):.2f}--{max(route_fps):.2f} analyzed updates/s, compared with {min(bypass_fps):.2f}--{max(bypass_fps):.2f} for route bypassed. GPU-board energy per analyzed update was {min(route_energy):.2f}--{max(route_energy):.2f}~J and {min(bypass_energy):.2f}--{max(bypass_energy):.2f}~J, respectively. These are configuration- and workload-specific board measurements, not total-system energy.

\begin{{table*}}[t]
\caption{{Same-pipeline 1440p replay on RTX~5090. Values are mean $\pm$ sample SD across three independent process runs over identical media. Calls/candidate and calls/update are per 100. Energy is GPU-board joules per analyzed update.}}
\label{{tab:same_pipeline_replay}}
\centering
\scriptsize
\begin{{tabular}}{{@{{}}llrrrrrr@{{}}}}
\toprule
Workload & Policy & Updates/s & Calls/cand. & Calls/update & p95 (ms) & Miss (\%) & J/update \\ \midrule
{chr(10).join(table_lines)}
\bottomrule
\end{{tabular}}
\end{{table*}}

\begin{{figure*}}[t]
\centering
\includegraphics[width=\textwidth]{{../figures/Fig6.pdf}}
\caption{{Same-pipeline replay. Bars are process means and error bars are sample SD ($n=3$). Both policies construct identical eligible candidates; only the M3 call decision differs.}}
\label{{fig:same_pipeline_replay}}
\end{{figure*}}
"""
    (manuscript / "same_pipeline_replay_results.tex").write_text(tex, encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    x = np.arange(3)
    width = 0.36
    workloads = ("normal", "mixed", "kinetic")
    for mode, offset, color in (("m3_route_on", -width/2, "#E65100"), ("m3_route_bypassed", width/2, "#6A1B9A")):
        r = [lookup[(mode, w)] for w in workloads]
        axes[0].bar(x + offset, [float(v["achieved_analysis_fps"]) for v in r], width,
                    yerr=[float(v["achieved_analysis_fps_sample_sd"]) for v in r], color=color, label=MODES[mode], capsize=3)
        axes[1].bar(x + offset, [100*float(v["q_candidate"]) for v in r], width,
                    yerr=[100*float(v["q_candidate_sample_sd"]) for v in r], color=color, capsize=3)
        axes[2].bar(x + offset, [float(v["gpu_board_energy_j_per_update"]) for v in r], width,
                    yerr=[float(v["gpu_board_energy_j_per_update_sample_sd"]) for v in r], color=color, capsize=3)
    axes[0].set_ylabel("Analyzed updates/s")
    axes[1].set_ylabel("Calls per 100 candidates")
    axes[2].set_ylabel("GPU-board J/update")
    for ax in axes:
        ax.set_xticks(x, [LABELS[w] for w in workloads], rotation=15)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "Fig6.pdf", bbox_inches="tight")
    plt.close(fig)

    (supplementary / "same_pipeline_replay_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (supplementary / "candidate_equivalence_audit.json").write_text(json.dumps(candidate_audit, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline-dir", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    manuscript = args.workspace / "manuscript"
    supplementary = args.workspace / "supplementary"
    figures = args.workspace / "figures"
    for path in (manuscript, supplementary, figures):
        path.mkdir(parents=True, exist_ok=True)
    offline = json.loads((args.offline_dir / "offline_policy_summary.json").read_text(encoding="utf-8"))
    predictions = read_predictions(args.offline_dir / "offline_policy_predictions.csv")
    replay = json.loads((args.replay_dir / "matched_replay_summary.json").read_text(encoding="utf-8"))
    audit = json.loads((args.replay_dir / "candidate_equivalence_audit.json").read_text(encoding="utf-8"))
    if not audit.get("audit_pass") or len(predictions) != 526:
        raise ValueError("sealed evidence gate failed")
    render_offline(offline, predictions, manuscript, supplementary, figures)
    render_replay(replay, audit, manuscript, supplementary, figures)


if __name__ == "__main__":
    main()
