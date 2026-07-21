#!/usr/bin/env python3
"""Generate source-wise frozen-prediction evidence for the v10 manuscript.

The script deliberately performs no inference. It joins every prediction file
to the canonical test manifest by video_id, validates labels and hashes, then
emits a long machine-readable table and an aggregation summary.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def binary_metrics(rows: Iterable[dict[str, int]]) -> dict[str, float | int]:
    counts = Counter()
    for row in rows:
        true = int(row["true_label"])
        pred = int(row["pred_label"])
        if true == 0 and pred == 0:
            counts["tn"] += 1
        elif true == 0 and pred == 1:
            counts["fp"] += 1
        elif true == 1 and pred == 0:
            counts["fn"] += 1
        elif true == 1 and pred == 1:
            counts["tp"] += 1
        else:
            raise ValueError(f"non-binary label/prediction: {true}, {pred}")

    tn, fp, fn, tp = (counts[k] for k in ("tn", "fp", "fn", "tp"))
    n = tn + fp + fn + tp
    accuracy = (tn + tp) / n if n else 0.0

    def f1(precision: float, recall: float) -> float:
        return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0

    precision_pos = tp / (tp + fp) if tp + fp else 0.0
    recall_pos = tp / (tp + fn) if tp + fn else 0.0
    precision_neg = tn / (tn + fn) if tn + fn else 0.0
    recall_neg = tn / (tn + fp) if tn + fp else 0.0
    f1_pos = f1(precision_pos, recall_pos)
    f1_neg = f1(precision_neg, recall_neg)
    return {
        "n": n,
        "correct": tn + tp,
        "accuracy": accuracy,
        "macro_f1": (f1_pos + f1_neg) / 2.0,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def validate_prediction_rows(
    rows: list[dict[str, str]],
    endpoint: str,
    test_manifest: dict[str, dict[str, str]],
    source_lookup: dict[str, str],
) -> list[dict[str, int | str]]:
    if not rows:
        raise ValueError(f"empty prediction file for {endpoint}")
    seen: set[str] = set()
    normalized: list[dict[str, int | str]] = []
    for raw in rows:
        video_id = str(raw["video_id"])
        if video_id in seen:
            raise ValueError(f"duplicate video_id in {endpoint}: {video_id}")
        seen.add(video_id)
        if video_id not in test_manifest:
            raise ValueError(f"{endpoint} contains non-test or unknown video_id: {video_id}")
        true_label = int(raw["true_label"] if "true_label" in raw else raw["label"])
        pred_label = int(raw["pred_label"])
        expected = int(test_manifest[video_id]["label"])
        if true_label != expected:
            raise ValueError(f"label mismatch for {endpoint}/{video_id}: {true_label} != {expected}")
        normalized.append(
            {
                "video_id": video_id,
                "source_dataset": source_lookup[video_id],
                "true_label": true_label,
                "pred_label": pred_label,
            }
        )
    expected_ids = set(test_manifest)
    if seen != expected_ids:
        missing = sorted(expected_ids - seen)
        extra = sorted(seen - expected_ids)
        raise ValueError(f"ID coverage mismatch for {endpoint}: missing={missing[:3]} extra={extra[:3]}")
    return normalized


def endpoint_rows(
    endpoint: str,
    seed: str,
    aggregation: str,
    normalized: list[dict[str, int | str]],
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, int]]] = defaultdict(list)
    for row in normalized:
        grouped[str(row["source_dataset"])].append(row)  # type: ignore[arg-type]
    result: list[dict[str, object]] = []
    for source in sorted(grouped):
        metrics = binary_metrics(grouped[source])
        result.append(
            {
                "endpoint": endpoint,
                "model_id": endpoint,
                "seed": seed,
                "source_dataset": source,
                "aggregation": aggregation,
                **metrics,
            }
        )
    return result


def external_rows(
    path: Path,
    test_manifest: dict[str, dict[str, str]],
    source_lookup: dict[str, str],
) -> tuple[list[dict[str, object]], dict[str, list[dict[str, object]]]]:
    rows = read_csv(path)
    grouped: dict[tuple[str, str, str], list[dict[str, int]]] = defaultdict(list)
    for raw in rows:
        model = str(raw["model_id"])
        seed = str(raw["seed"])
        video_id = str(raw["video_id"])
        if video_id not in test_manifest:
            raise ValueError(f"external prediction contains unknown video_id: {video_id}")
        true_label = int(raw["true_label"])
        pred_label = int(raw["pred_label"])
        if true_label != int(test_manifest[video_id]["label"]):
            raise ValueError(f"external label mismatch for {model}/{seed}/{video_id}")
        grouped[(model, seed, source_lookup[video_id])].append(
            {"true_label": true_label, "pred_label": pred_label}
        )

    raw_rows: list[dict[str, object]] = []
    for (model, seed, source), values in sorted(grouped.items()):
        raw_rows.append(
            {
                "endpoint": model,
                "model_id": model,
                "seed": seed,
                "source_dataset": source,
                "aggregation": "seed_point",
                **binary_metrics(values),
            }
        )
    by_source_model: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in raw_rows:
        by_source_model[f"{row['model_id']}::{row['source_dataset']}"].append(row)
    return raw_rows, by_source_model


def aggregate_external(raw_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in raw_rows:
        grouped[(str(row["model_id"]), str(row["source_dataset"]))].append(row)
    output: list[dict[str, object]] = []
    for (model, source), values in sorted(grouped.items()):
        if len(values) != 3 or sorted(str(v["seed"]) for v in values) != ["50900", "50901", "50902"]:
            raise ValueError(f"external endpoint must have exactly seeds 50900-50902: {model}/{source}")
        accuracies = [float(v["accuracy"]) for v in values]
        f1s = [float(v["macro_f1"]) for v in values]
        mean_acc = sum(accuracies) / len(accuracies)
        mean_f1 = sum(f1s) / len(f1s)
        sd_acc = (sum((x - mean_acc) ** 2 for x in accuracies) / (len(accuracies) - 1)) ** 0.5
        sd_f1 = (sum((x - mean_f1) ** 2 for x in f1s) / (len(f1s) - 1)) ** 0.5
        output.append(
            {
                "model_id": model,
                "source_dataset": source,
                "n": values[0]["n"],
                "accuracy_mean": mean_acc,
                "accuracy_sample_sd": sd_acc,
                "macro_f1_mean": mean_f1,
                "macro_f1_sample_sd": sd_f1,
                "seeds": "50900;50901;50902",
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def latex_number(value: object, digits: int = 3) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def latex_percent(value: object) -> str:
    return f"{100.0 * float(value):.2f}\\%"


def write_supplementary_tex(
    path: Path,
    source_counts: dict[str, dict[str, int]],
    all_rows: list[dict[str, object]],
    external_summary: list[dict[str, object]],
) -> None:
    sources = sorted(source_counts)
    source_labels = {
        "hockey": "Hockey Fights",
        "movies2": "Real-Life Violence",
        "rwf2000": "RWF-2000",
        "surv_fight": "Surveillance Fight",
        "violent_flows": "Violent Flows",
    }
    endpoint_labels = {
        "m1_dense": "Dense whole frame",
        "m3_crowd": "Crowd-centric",
        "routed_offline": "Offline routed",
        "c3d": "C3D",
        "i3d": "I3D",
        "resnet_lstm": "ResNet--LSTM",
        "slowfast": "SlowFast",
        "swin3d": "Swin3D",
        "josenet": "JOSENet",
    }
    lines = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[a4paper,margin=22mm]{geometry}",
        r"\usepackage{booktabs,longtable,tabularx,array}",
        r"\newcolumntype{Y}{>{\centering\arraybackslash}X}",
        r"\renewcommand{\thetable}{S\arabic{table}}",
        r"\title{Supplementary Information for\\\large Conditional Routing for 2K Violence Detection: Accuracy, Invocation Cost, and Cross-Device Evaluation}",
        r"\author{Duc Viet Vu}",
        r"\date{}",
        r"\begin{document}",
        r"\maketitle",
        r"\noindent\textbf{Scope.} These tables are descriptive post hoc aggregations of frozen test predictions. Source assignment uses the canonical test manifest and \texttt{video\_id}; no inference or threshold tuning was performed for this analysis.",
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Canonical test-cohort composition by source.}",
        r"\label{tab:sourcewise-cohort}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Source dataset & $n$ & Non-violence & Violence\\",
        r"\midrule",
    ]
    for source in sources:
        item = source_counts[source]
        lines.append(f"{source_labels[source]} & {item['n']} & {item['non_violence']} & {item['violence']}\\\\")
    lines += [
        r"\midrule",
        f"Total & {sum(v['n'] for v in source_counts.values())} & {sum(v['non_violence'] for v in source_counts.values())} & {sum(v['violence'] for v in source_counts.values())}\\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        r"\begin{longtable}{llrrrl}",
        r"\caption{Source-wise frozen-prediction results. Locked endpoints are point estimates with TN/FP/FN/TP; external baselines are mean $\pm$ sample SD across seeds 50900--50902, with per-seed confusion counts in the accompanying CSV.}\label{tab:sourcewise-results}\\",
        r"\toprule",
        r"Endpoint & Source & $n$ & Accuracy & Macro-F1 & TN/FP/FN/TP\\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Endpoint & Source & $n$ & Accuracy & Macro-F1 & TN/FP/FN/TP\\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ]
    endpoint_order = {"m1_dense": 0, "m3_crowd": 1, "routed_offline": 2, "c3d": 3, "i3d": 4, "resnet_lstm": 5, "slowfast": 6, "swin3d": 7, "josenet": 8}
    locked_rows = [row for row in all_rows if str(row["aggregation"]) == "locked"]
    for row in sorted(locked_rows, key=lambda r: (endpoint_order.get(str(r["endpoint"]), 99), str(r["source_dataset"]))):
        endpoint = endpoint_labels[str(row["endpoint"])]
        source = source_labels[str(row["source_dataset"])]
        confusion = f"{row['tn']}/{row['fp']}/{row['fn']}/{row['tp']}"
        lines.append(
            f"{endpoint} & {source} & {row['n']} & {latex_percent(row['accuracy'])} & {latex_number(row['macro_f1'], 4)} & {confusion}\\\\"
        )
    lines.append(r"\midrule")
    for row in sorted(external_summary, key=lambda r: (endpoint_order.get(str(r["model_id"]), 99), str(r["source_dataset"]))):
        endpoint = endpoint_labels[str(row["model_id"])]
        source = source_labels[str(row["source_dataset"])]
        accuracy = f"{100*float(row['accuracy_mean']):.2f} $\\pm$ {100*float(row['accuracy_sample_sd']):.2f}\\%"
        macro_f1 = f"{float(row['macro_f1_mean']):.4f} $\\pm$ {float(row['macro_f1_sample_sd']):.4f}"
        lines.append(f"{endpoint} & {source} & {row['n']} & {accuracy} & {macro_f1} & see CSV\\\\")
    lines += [
        r"\end{longtable}",
        r"\noindent For the six external baselines, the manuscript reports mean $\pm$ sample SD across seeds 50900--50902; the machine-readable CSV preserves the per-seed rows and all confusion counts. Small strata, especially RWF-2000, are not interpreted as independent estimates of external generalization.",
        r"\end{document}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--m1", type=Path, required=True)
    parser.add_argument("--m3", type=Path, required=True)
    parser.add_argument("--routed", type=Path, required=True)
    parser.add_argument("--external", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = read_csv(args.manifest)
    test_rows = [r for r in manifest_rows if r.get("new_split") == "test"]
    test_manifest = {str(r["video_id"]): r for r in test_rows}
    if len(test_manifest) != 526:
        raise ValueError(f"expected 526 canonical test IDs, found {len(test_manifest)}")
    source_lookup = {video_id: str(row["source_dataset"]) for video_id, row in test_manifest.items()}
    source_counts: dict[str, dict[str, int]] = {}
    for source in sorted(set(source_lookup.values())):
        rows = [r for r in test_rows if r["source_dataset"] == source]
        source_counts[source] = {
            "n": len(rows),
            "non_violence": sum(int(r["label"]) == 0 for r in rows),
            "violence": sum(int(r["label"]) == 1 for r in rows),
        }

    locked_specs = [
        ("m1_dense", "50900", "locked", args.m1),
        ("m3_crowd", "50902", "locked", args.m3),
        ("routed_offline", "50902", "locked", args.routed),
    ]
    all_rows: list[dict[str, object]] = []
    input_hashes = {"manifest": sha256(args.manifest), "external": sha256(args.external)}
    overall: dict[str, dict[str, object]] = {}
    for endpoint, seed, aggregation, path in locked_specs:
        input_hashes[endpoint] = sha256(path)
        normalized = validate_prediction_rows(read_csv(path), endpoint, test_manifest, source_lookup)
        all_rows.extend(endpoint_rows(endpoint, seed, aggregation, normalized))
        overall[endpoint] = binary_metrics(normalized)  # type: ignore[arg-type]

    external_raw, _ = external_rows(args.external, test_manifest, source_lookup)
    all_rows.extend(external_raw)
    external_models = sorted({str(r["model_id"]) for r in external_raw})
    if external_models != ["c3d", "i3d", "josenet", "resnet_lstm", "slowfast", "swin3d"]:
        raise ValueError(f"unexpected external model set: {external_models}")
    external_summary = aggregate_external(external_raw)

    raw_fields = [
        "endpoint", "model_id", "seed", "source_dataset", "aggregation", "n", "correct",
        "accuracy", "macro_f1", "tn", "fp", "fn", "tp",
    ]
    write_csv(args.output_dir / "sourcewise_results.csv", all_rows, raw_fields)
    write_csv(
        args.output_dir / "sourcewise_external_summary.csv",
        external_summary,
        ["model_id", "source_dataset", "n", "accuracy_mean", "accuracy_sample_sd", "macro_f1_mean", "macro_f1_sample_sd", "seeds"],
    )
    write_csv(
        args.output_dir / "sourcewise_cohort.csv",
        [{"source_dataset": source, **values} for source, values in sorted(source_counts.items())],
        ["source_dataset", "n", "non_violence", "violence"],
    )
    summary = {
        "schema_version": "sourcewise_v10_v1",
        "analysis_type": "post hoc aggregation of frozen predictions; no inference",
        "canonical_test_n": len(test_manifest),
        "source_counts": source_counts,
        "input_sha256": input_hashes,
        "locked_endpoints": ["m1_dense", "m3_crowd", "routed_offline"],
        "locked_checkpoint_seeds": {"m1_dense": 50900, "m3_crowd": 50902, "routed_offline": 50902},
        "external_models": external_models,
        "external_seed_rule": [50900, 50901, 50902],
        "aggregation_rule": {
            "locked": "single frozen checkpoint point estimate per source",
            "external": "mean and sample SD across exactly three independent seeds per source",
            "metrics": "binary accuracy, macro-F1, and TN/FP/FN/TP; source assigned by canonical manifest video_id",
        },
        "overall_locked_endpoints": overall,
        "files": {
            "sourcewise_results_csv": "sourcewise_results.csv",
            "sourcewise_external_summary_csv": "sourcewise_external_summary.csv",
            "sourcewise_cohort_csv": "sourcewise_cohort.csv",
        },
        "pre_amendment_artifacts_excluded": [
            "pipeline_artifacts/stage4_prime_analysis/paired_v2_seed50900/stage4_prime_paired_accuracy_analysis.json"
        ],
    }
    (args.output_dir / "sourcewise_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_supplementary_tex(args.output_dir / "supplementary_sourcewise.tex", source_counts, all_rows, external_summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
