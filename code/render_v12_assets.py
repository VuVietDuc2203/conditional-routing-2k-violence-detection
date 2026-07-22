#!/usr/bin/env python3
"""Render JVCIR v12 manuscript snippets, figures, and supplementary tables."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


WORKLOADS = ("normal", "mixed", "kinetic")
WORKLOAD_LABELS = {"normal": "Normal", "mixed": "Mixed", "kinetic": "Kinetic-rich"}
MODES = ("m3_route_on", "m3_route_bypassed")
MODE_LABELS = {"m3_route_on": "Route on", "m3_route_bypassed": "Route bypassed"}
COLORS = {"m3_route_on": "#0077BB", "m3_route_bypassed": "#EE7733"}
HATCHES = {"m3_route_on": "", "m3_route_bypassed": "///"}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def draw_box(ax, x, y, width, height, text, color, fontsize=8, linewidth=1.0):
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        facecolor=color, edgecolor="#222222", linewidth=linewidth,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=fontsize)


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10, linewidth=1.0, color="#333333"))


def render_figure1(figures: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.8, 5.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.94, "A  Frozen-prediction video counterfactual", weight="bold", fontsize=11)
    ax.text(0.02, 0.48, "B  Matched same-pipeline replay", weight="bold", fontsize=11)

    boxes_a = [
        (0.03, 0.69, 0.16, 0.13, "526 test videos\n+ frozen labels", "#E9F2FA"),
        (0.25, 0.78, 0.20, 0.10, "Frozen M3\nscore + prediction", "#D4E8F6"),
        (0.25, 0.59, 0.20, 0.10, "Frozen router artifact\ncall / skip only", "#E3F3EE"),
        (0.53, 0.69, 0.19, 0.13, "Call/skip combination\nsame M3 if called", "#E8F3E9"),
        (0.79, 0.69, 0.18, 0.13, "Video-level endpoints\naccuracy + $q_{video}$", "#FFF0D9"),
    ]
    for item in boxes_a:
        draw_box(ax, *item)
    arrow(ax, (0.19, 0.755), (0.25, 0.83))
    arrow(ax, (0.19, 0.755), (0.25, 0.64))
    arrow(ax, (0.45, 0.83), (0.53, 0.77))
    arrow(ax, (0.45, 0.64), (0.53, 0.73))
    arrow(ax, (0.72, 0.755), (0.79, 0.755))
    ax.text(0.50, 0.61, "Join by video ID; no inference rerun and no candidate-equivalence claim", ha="center", fontsize=8, color="#555555")

    boxes_b = [
        (0.03, 0.22, 0.16, 0.13, "1440p / 30-FPS\nsource", "#E9F2FA"),
        (0.23, 0.22, 0.22, 0.13, "8-Hz detector + tracker\n+ crowd state + T50 crop", "#D4E8F6"),
        (0.49, 0.22, 0.17, 0.13, "Candidate ID +\ntensor SHA-256", "#E3F3EE"),
        (0.70, 0.31, 0.17, 0.10, "Route on\nconditional call", "#D9ECF7"),
        (0.70, 0.15, 0.17, 0.10, "Route bypassed\nalways call candidate", "#FCE5D5"),
        (0.91, 0.22, 0.07, 0.13, "Same\nM3", "#FFF0D9"),
    ]
    for item in boxes_b:
        draw_box(ax, *item, fontsize=7.5)
    arrow(ax, (0.19, 0.285), (0.23, 0.285))
    arrow(ax, (0.45, 0.285), (0.49, 0.285))
    arrow(ax, (0.66, 0.285), (0.70, 0.36))
    arrow(ax, (0.66, 0.285), (0.70, 0.20))
    arrow(ax, (0.87, 0.36), (0.91, 0.31))
    arrow(ax, (0.87, 0.20), (0.91, 0.25))
    ax.text(0.50, 0.07, "Candidate equivalence is audited only on the replay track (9/9 workload-repeat pairs).", ha="center", fontsize=8, color="#555555")
    fig.tight_layout()
    save(fig, figures / "Fig1")


def render_replay_figure(replay: dict, figures: Path) -> None:
    rows = {(row["mode"], row["workload"]): row for row in replay["aggregate_rows"]}
    specifications = [
        ("q_candidate", "Calls per 100 candidates", 100.0, "(a) Candidate calls"),
        ("cpu_percent_mean", "Process CPU (%)", 1.0, "(b) Process CPU"),
        ("achieved_analysis_fps", "Analyzed updates/s", 1.0, "(c) End-to-end throughput"),
        ("gpu_board_energy_j_per_update", "GPU-board J/update", 1.0, "(d) Board energy"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.1))
    x = np.arange(len(WORKLOADS))
    width = 0.36
    for ax, (metric, ylabel, scale, title) in zip(axes.flat, specifications):
        for index, mode in enumerate(MODES):
            mode_rows = [rows[(mode, workload)] for workload in WORKLOADS]
            means = [scale * float(row[metric]) for row in mode_rows]
            sds = [scale * float(row[f"{metric}_sample_sd"]) for row in mode_rows]
            ax.bar(
                x + (index - 0.5) * width, means, width, yerr=sds, capsize=3,
                color=COLORS[mode], hatch=HATCHES[mode], edgecolor="black", linewidth=0.5,
                label=MODE_LABELS[mode],
            )
        ax.set_xticks(x, [WORKLOAD_LABELS[w] for w in WORKLOADS], rotation=10)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", weight="bold")
        ax.grid(axis="y", color="#dddddd", linewidth=0.5)
        ax.set_axisbelow(True)
    axes[1, 0].axhline(8.0, linestyle="--", color="#333333", linewidth=1, label="8-Hz target")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, figures / "Fig6")


def render_counterfactual_figure(counter: dict, figures: Path) -> None:
    categories = counter["paired_outcome_categories"]
    labels = ["Both\ncorrect", "Route only\ncorrect", "Bypass only\ncorrect", "Both\nwrong"]
    keys = ["both_correct", "route_on_only_correct", "bypass_only_correct", "both_wrong"]
    colors = ["#009988", "#33BBEE", "#EE7733", "#BBBBBB"]
    values = [categories[key] for key in keys]
    fig, ax = plt.subplots(figsize=(6.9, 4.2))
    bars = ax.bar(np.arange(4), values, color=colors, edgecolor="black", linewidth=0.5)
    ax.bar_label(bars, padding=3)
    ax.set_xticks(np.arange(4), labels)
    ax.set_ylabel("Held-out videos")
    ax.set_title("Paired outcomes from frozen M3 predictions")
    fig.tight_layout()
    save(fig, figures / "Fig7")


def render_graphical_abstract(counter: dict, replay: dict, figures: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.8, 4.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.93, "KINEMATIC ROUTING: ACCURACY--INVOCATION AND RUNTIME EVIDENCE", ha="center", weight="bold", fontsize=13)
    draw_box(ax, 0.03, 0.20, 0.28, 0.58, "FROZEN VIDEO COUNTERFACTUAL\n\nBypass: 502/526 (95.44%)\nRoute on: 494/526 (93.92%)\nCalls: 489/526 (92.97%)\nAccuracy change: -1.52 points", "#E9F2FA", fontsize=10)
    draw_box(ax, 0.36, 0.20, 0.28, 0.58, "SAME-PIPELINE REPLAY\n\nIdentical candidates: 9/9 pairs\nCandidate calls: -77.7% to -78.5%\nProcess CPU: about half\nRTX 5090, three workloads, n=3", "#E3F3EE", fontsize=10)
    draw_box(ax, 0.69, 0.20, 0.28, 0.58, "BOUNDED CONCLUSION\n\nThroughput and deadline misses\ndid not improve consistently.\nBoard-energy changes were modest\nand workload-specific.", "#FFF0D9", fontsize=10)
    arrow(ax, (0.31, 0.49), (0.36, 0.49))
    arrow(ax, (0.64, 0.49), (0.69, 0.49))
    ax.text(0.5, 0.08, "Offline recognition and operational replay are separate endpoints.", ha="center", fontsize=9, color="#444444")
    fig.tight_layout()
    save(fig, figures / "graphical_abstract")


def mean_sd(row: dict, metric: str, scale: float = 1.0, digits: int = 2) -> str:
    return f"{scale*float(row[metric]):.{digits}f} $\\pm$ {scale*float(row[f'{metric}_sample_sd']):.{digits}f}"


def render_snippets(counter: dict, replay: dict, manuscript: Path) -> None:
    boot = counter["semantic_group_bootstrap"]["accuracy_difference_route_on_minus_bypass"]["ci95_percentile"]
    mcnemar = counter["paired_mcnemar"]
    bypass, route = counter["bypass"], counter["route_on"]
    offline = rf"""The frozen-prediction counterfactual invoked M3 for 489/526 videos ($q_{{\mathrm{{video}}}}=92.97\%$). Bypass reused the frozen M3 prediction for every video and obtained {bypass['correct']}/526 ({100*bypass['accuracy']:.2f}\%) accuracy with macro-F1 {100*bypass['macro_f1']:.2f}\%. Route on reused that same prediction for invoked videos and emitted the predefined normal label otherwise, obtaining {route['correct']}/526 ({100*route['accuracy']:.2f}\%) accuracy with macro-F1 {100*route['macro_f1']:.2f}\%. The paired difference was {100*(route['accuracy']-bypass['accuracy']):.2f} percentage points (95\% semantic-group bootstrap interval {100*boot[0]:.2f} to {100*boot[1]:.2f}); exact McNemar discordance was {mcnemar['route_on_only_correct']} versus {mcnemar['bypass_only_correct']} videos ($p={mcnemar['exact_two_sided_p']:.4f}$).

\begin{{table}}[t]
\caption{{Frozen-prediction video-level call/skip counterfactual on 526 held-out videos. Both policies reuse the same frozen M3 prediction when the classifier is invoked.}}
\label{{tab:frozen_counterfactual}}
\centering
\small
\begin{{tabular}}{{@{{}}lrrrrrr@{{}}}}
\toprule
Policy & Accuracy & Macro-F1 & TN & FP & FN & TP \\ \midrule
Route on & {100*route['accuracy']:.2f}\% & {100*route['macro_f1']:.2f}\% & {route['tn']} & {route['fp']} & {route['fn']} & {route['tp']} \\
Bypass & {100*bypass['accuracy']:.2f}\% & {100*bypass['macro_f1']:.2f}\% & {bypass['tn']} & {bypass['fp']} & {bypass['fn']} & {bypass['tp']} \\ \bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[t]
\centering
\includegraphics[width=\columnwidth]{{../figures/Fig7.pdf}}
\caption{{Paired correctness categories for the frozen-prediction counterfactual. The unit is a held-out video; the comparison does not assert candidate-level implementation equivalence.}}
\label{{fig:frozen_counterfactual}}
\end{{figure}}
"""
    (manuscript / "frozen_counterfactual_results.tex").write_text(offline, encoding="utf-8")

    rows = {(row["mode"], row["workload"]): row for row in replay["aggregate_rows"]}
    lines = []
    for workload in WORKLOADS:
        for mode in MODES:
            row = rows[(mode, workload)]
            lines.append(
                f"{WORKLOAD_LABELS[workload]} & {MODE_LABELS[mode]} & {mean_sd(row, 'q_candidate', 100)} & "
                f"{mean_sd(row, 'cpu_percent_mean')} & {mean_sd(row, 'achieved_analysis_fps')} & "
                f"{mean_sd(row, 'latency_p95_ms')} & {mean_sd(row, 'deadline_miss_rate', 100)} & "
                f"{mean_sd(row, 'gpu_board_energy_j_per_update')} \\\\"
            )
    replay_tex = rf"""Candidate IDs, tensor SHA-256 values, counts, and source times matched in all nine workload--repeat pairs. Route on classified 21.52--22.29\% of eligible candidates, a 77.71--78.48\% reduction from bypass. Mean process CPU was 49.07--52.40\% lower across workloads. Throughput was lower on normal, slightly higher on mixed, and nearly unchanged on kinetic-rich replay; deadline-miss rates were not consistently lower. Mean GPU-board energy was lower for route on, but paired variation was substantial on the normal workload. These board measurements do not represent total-system energy.

\begin{{table*}}[t]
\caption{{Matched same-pipeline 1440p replay on RTX~5090. Values are mean $\pm$ sample SD across three independent processes over identical media. Calls/candidate and misses are percentages; energy is GPU-board joules per analyzed update.}}
\label{{tab:same_pipeline_replay}}
\centering
\scriptsize
\begin{{tabular}}{{@{{}}llrrrrrr@{{}}}}
\toprule
Workload & Policy & Calls/cand. & CPU (\%) & Updates/s & p95 (ms) & Miss (\%) & J/update \\ \midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}
\end{{table*}}

\begin{{figure*}}[t]
\centering
\includegraphics[width=0.92\textwidth]{{../figures/Fig6.pdf}}
\caption{{Same-pipeline replay. Bars are process means and error bars are sample SD ($n=3$). Candidate calls and process CPU fall under routing, whereas throughput, deadline behavior, and board energy remain workload-dependent.}}
\label{{fig:same_pipeline_replay}}
\end{{figure*}}
"""
    (manuscript / "same_pipeline_replay_results_v12.tex").write_text(replay_tex, encoding="utf-8")


def render_supplement(counter_dir: Path, replay_dir: Path, output: Path) -> None:
    source_rows = list(csv.DictReader((counter_dir / "frozen_counterfactual_sourcewise.csv").open(encoding="utf-8", newline="")))
    stage_rows = list(csv.DictReader((replay_dir / "replay_stage_summary.csv").open(encoding="utf-8", newline="")))
    source_lines = []
    for row in source_rows:
        source_lines.append(
            f"{row['source_dataset'].replace('_',' ')} & {row['policy'].replace('_',' ')} & {row['n']} & "
            f"{100*float(row['accuracy']):.2f} & {100*float(row['macro_f1']):.2f} & "
            f"{row['tn']} & {row['fp']} & {row['fn']} & {row['tp']} \\\\"
        )
    stage_lines = []
    for row in stage_rows:
        stage_lines.append(
            f"{WORKLOAD_LABELS[row['workload']]} & {MODE_LABELS[row['mode']]} & "
            f"{float(row['total_ms_mean']):.2f} $\\pm$ {float(row['total_ms_sample_sd']):.2f} & "
            f"{float(row['yolo_ms_mean']):.2f} $\\pm$ {float(row['yolo_ms_sample_sd']):.2f} & "
            f"{float(row['hdbscan_ms_mean']):.2f} $\\pm$ {float(row['hdbscan_ms_sample_sd']):.2f} & "
            f"{float(row['gate_ms_mean']):.2f} $\\pm$ {float(row['gate_ms_sample_sd']):.2f} & "
            f"{float(row['crop_ms_mean']):.2f} $\\pm$ {float(row['crop_ms_sample_sd']):.2f} & "
            f"{float(row['classifier_ms_mean']):.2f} $\\pm$ {float(row['classifier_ms_sample_sd']):.2f} \\\\"
        )
    tex = rf"""\documentclass[9pt]{{article}}
\usepackage[a4paper,margin=14mm]{{geometry}}
\usepackage{{booktabs}}
\title{{Supplementary Material: Kinematic Routing for Violence Recognition}}
\author{{Duc Viet Vu}}
\date{{}}
\begin{{document}}
\maketitle
\section*{{S1. Frozen-prediction source-wise results}}
Source strata are descriptive and do not estimate external-domain generalization.
\begin{{table}}[h]
\centering\scriptsize
\begin{{tabular}}{{@{{}}llrrrrrrr@{{}}}}
\toprule
Source & Policy & $n$ & Acc. (\%) & Macro-F1 (\%) & TN & FP & FN & TP \\ \midrule
{chr(10).join(source_lines)}
\bottomrule
\end{{tabular}}
\end{{table}}

\section*{{S2. Replay stage timing}}
Values are mean $\pm$ sample SD across three process means. Columns are descriptive and not additive: YOLO inference is contained in detector/tracker time, and one update may contain multiple classifier calls.
\begin{{table}}[h]
\centering\scriptsize
\begin{{tabular}}{{@{{}}llrrrrrr@{{}}}}
\toprule
Workload & Policy & Total & YOLO/tracker & HDBSCAN & Gate & Crop & Classifier \\ \midrule
{chr(10).join(stage_lines)}
\bottomrule
\end{{tabular}}
\end{{table}}
\end{{document}}
"""
    output.write_text(tex, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace
    counter_dir = workspace / "evidence" / "counterfactual"
    replay_dir = workspace / "evidence" / "replay_v12"
    counter = load(counter_dir / "frozen_counterfactual_summary.json")
    replay = load(replay_dir / "replay_v12_summary.json")
    figures = workspace / "figures"
    manuscript = workspace / "manuscript"
    figures.mkdir(parents=True, exist_ok=True)
    setup_style()
    render_figure1(figures)
    render_replay_figure(replay, figures)
    render_counterfactual_figure(counter, figures)
    render_graphical_abstract(counter, replay, figures)
    render_snippets(counter, replay, manuscript)
    render_supplement(counter_dir, replay_dir, workspace / "supplementary" / "supplementary_jvcir_v12.tex")


if __name__ == "__main__":
    main()
