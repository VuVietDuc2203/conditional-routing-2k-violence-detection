#!/usr/bin/env python3
"""Build the JVCIR v12 video-level call/skip counterfactual from frozen files."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path


EXPECTED_M3_SHA256 = "a77f0a0d2a7c768c9ee0ecad854bb78c578fc435db0fabd0e3520f4060a68f29"
EXPECTED_ROUTER_SHA256 = "9d4775123fc4c6bae648087e5fe77386b99aadc8f0692ffc16b7f712fe3d0845"
INVOKED_STATUSES = {"success", "success_full_clip_on_gate"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def indexed(rows: list[dict[str, str]], label: str) -> dict[str, dict[str, str]]:
    result = {row["video_id"]: row for row in rows}
    if len(rows) != 526 or len(result) != 526:
        raise ValueError(f"{label}: expected 526 unique video IDs, found {len(rows)}/{len(result)}")
    return result


def metrics(rows: list[dict[str, object]], pred_key: str) -> dict[str, object]:
    tn = fp = fn = tp = 0
    for row in rows:
        y, pred = int(row["true_label"]), int(row[pred_key])
        if y == 0 and pred == 0:
            tn += 1
        elif y == 0 and pred == 1:
            fp += 1
        elif y == 1 and pred == 0:
            fn += 1
        else:
            tp += 1
    f1_normal = 2 * tn / max(1, 2 * tn + fp + fn)
    f1_violence = 2 * tp / max(1, 2 * tp + fp + fn)
    return {
        "n": len(rows),
        "correct": tn + tp,
        "accuracy": (tn + tp) / len(rows),
        "macro_f1": 0.5 * (f1_normal + f1_violence),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def exact_mcnemar(rows: list[dict[str, object]]) -> dict[str, object]:
    route_only = sum(
        int(row["route_on_pred"]) == int(row["true_label"])
        and int(row["bypass_pred"]) != int(row["true_label"])
        for row in rows
    )
    bypass_only = sum(
        int(row["bypass_pred"]) == int(row["true_label"])
        and int(row["route_on_pred"]) != int(row["true_label"])
        for row in rows
    )
    discordant = route_only + bypass_only
    if discordant == 0:
        p_value = 1.0
    else:
        lower = min(route_only, bypass_only)
        p_value = min(
            1.0,
            2.0 * sum(math.comb(discordant, k) for k in range(lower + 1)) / (2**discordant),
        )
    return {
        "route_on_only_correct": route_only,
        "bypass_only_correct": bypass_only,
        "discordant": discordant,
        "exact_two_sided_p": p_value,
    }


def quantile(values: list[float], q: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return values[low]
    return values[low] * (high - position) + values[high] * (position - low)


def grouped_bootstrap(rows: list[dict[str, object]], replicates: int, seed: int) -> dict[str, object]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row["semantic_group_id"])].append(row)
    keys = sorted(groups)
    rng = random.Random(seed)
    differences: list[float] = []
    for _ in range(replicates):
        sample: list[dict[str, object]] = []
        for _ in keys:
            sample.extend(groups[rng.choice(keys)])
        route_accuracy = float(metrics(sample, "route_on_pred")["accuracy"])
        bypass_accuracy = float(metrics(sample, "bypass_pred")["accuracy"])
        differences.append(route_accuracy - bypass_accuracy)
    return {
        "replicates": replicates,
        "seed": seed,
        "resampling_unit": "semantic_group_id",
        "group_count": len(keys),
        "accuracy_difference_route_on_minus_bypass": {
            "mean": sum(differences) / len(differences),
            "ci95_percentile": [quantile(differences, 0.025), quantile(differences, 0.975)],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m3-predictions", type=Path, required=True)
    parser.add_argument("--router-predictions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=50900)
    args = parser.parse_args()

    input_hashes = {
        "m3_predictions_sha256": sha256(args.m3_predictions),
        "router_predictions_sha256": sha256(args.router_predictions),
        "manifest_sha256": sha256(args.manifest),
    }
    if input_hashes["m3_predictions_sha256"] != EXPECTED_M3_SHA256:
        raise ValueError("frozen M3 prediction hash mismatch")
    if input_hashes["router_predictions_sha256"] != EXPECTED_ROUTER_SHA256:
        raise ValueError("frozen router prediction hash mismatch")

    m3 = indexed(read_csv(args.m3_predictions), "M3 predictions")
    router = indexed(read_csv(args.router_predictions), "router predictions")
    manifest_rows = [row for row in read_csv(args.manifest) if row.get("split") == "test"]
    manifest = indexed(manifest_rows, "test manifest")
    if set(m3) != set(router) or set(m3) != set(manifest):
        raise ValueError("video-ID sets differ across frozen inputs")

    output_rows: list[dict[str, object]] = []
    for item in manifest_rows:
        video_id = item["video_id"]
        dense, routed = m3[video_id], router[video_id]
        labels = {int(item["label"]), int(dense["true_label"]), int(routed["true_label"])}
        if len(labels) != 1:
            raise ValueError(f"label disagreement for {video_id}")
        invoked = routed["preprocess_status"] in INVOKED_STATUSES
        bypass_pred = int(dense["pred_label"])
        route_pred = bypass_pred if invoked else 0
        y = int(item["label"])
        output_rows.append({
            "video_id": video_id,
            "semantic_group_id": item["semantic_group_id"],
            "source_dataset": item["source_dataset"],
            "true_label": y,
            "classifier_invoked": int(invoked),
            "router_status": routed["preprocess_status"],
            "frozen_m3_score": float(dense["score_violence"]),
            "bypass_pred": bypass_pred,
            "route_on_pred": route_pred,
            "bypass_correct": int(bypass_pred == y),
            "route_on_correct": int(route_pred == y),
        })

    bypass = metrics(output_rows, "bypass_pred")
    route = metrics(output_rows, "route_on_pred")
    invoked_count = sum(int(row["classifier_invoked"]) for row in output_rows)
    if (bypass["correct"], route["correct"], invoked_count) != (502, 494, 489):
        raise ValueError("frozen endpoint headline values do not match the locked protocol")

    sourcewise: list[dict[str, object]] = []
    for source in sorted({str(row["source_dataset"]) for row in output_rows}):
        subset = [row for row in output_rows if row["source_dataset"] == source]
        for policy, key in (("route_on", "route_on_pred"), ("bypass", "bypass_pred")):
            result = metrics(subset, key)
            sourcewise.append({"source_dataset": source, "policy": policy, **result})

    outcome_categories = {
        "both_correct": sum(row["bypass_correct"] and row["route_on_correct"] for row in output_rows),
        "route_on_only_correct": sum((not row["bypass_correct"]) and row["route_on_correct"] for row in output_rows),
        "bypass_only_correct": sum(row["bypass_correct"] and (not row["route_on_correct"]) for row in output_rows),
        "both_wrong": sum((not row["bypass_correct"]) and (not row["route_on_correct"]) for row in output_rows),
    }
    summary = {
        "protocol": "jvcir_v12_frozen_prediction_video_call_skip_counterfactual",
        "endpoint_boundary": "video-level post-hoc counterfactual; no re-inference and no same-pipeline runtime equivalence",
        "skipped_prediction_rule": "predefined normal label (0)",
        "invoked_statuses": sorted(INVOKED_STATUSES),
        "input_hashes": input_hashes,
        "bypass": bypass,
        "route_on": route,
        "invocation": {
            "invoked_videos": invoked_count,
            "total_videos": len(output_rows),
            "q_video": invoked_count / len(output_rows),
        },
        "accuracy_difference_route_on_minus_bypass": route["accuracy"] - bypass["accuracy"],
        "paired_mcnemar": exact_mcnemar(output_rows),
        "semantic_group_bootstrap": grouped_bootstrap(output_rows, args.bootstrap, args.seed),
        "paired_outcome_categories": outcome_categories,
        "sourcewise_interpretation": "descriptive only; source strata are unequal and do not estimate external-domain generalization",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / "frozen_counterfactual_predictions.csv"
    with prediction_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    sourcewise_path = args.output_dir / "frozen_counterfactual_sourcewise.csv"
    with sourcewise_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sourcewise[0]))
        writer.writeheader()
        writer.writerows(sourcewise)
    summary_path = args.output_dir / "frozen_counterfactual_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    completion = {
        "status": "complete",
        "protocol": summary["protocol"],
        "row_count": len(output_rows),
        "input_hashes": input_hashes,
        "predictions_sha256": sha256(prediction_path),
        "sourcewise_sha256": sha256(sourcewise_path),
        "summary_sha256": sha256(summary_path),
    }
    (args.output_dir / "FROZEN_COUNTERFACTUAL_COMPLETE.json").write_text(
        json.dumps(completion, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

