#!/usr/bin/env python3
"""Paired route-on versus route-bypassed evaluation on the frozen 526-video cohort.

The runtime materializes one common M3 candidate stream and executes the
bypassed policy. Route-on predictions are then derived only from calls whose
precomputed route decision is true, so candidate construction cannot differ
between the paired endpoints.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from data.scripts.build_movinet_preprocess_cache import make_clip_tensor
from training_code.m3_streaming_runtime import M3StreamingRuntime, RuntimeConfig


FIELDS = [
    "video_id", "semantic_group_id", "source_dataset", "source_video", "true_label",
    "route_on_pred", "route_on_score", "route_bypassed_pred", "route_bypassed_score",
    "eligible_candidates", "route_selected_candidates", "candidate_ids", "candidate_sha256s",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("split") == "test"]
    if len(rows) != 526 or len({row["video_id"] for row in rows}) != 526:
        raise ValueError(f"expected 526 unique test rows, found {len(rows)}")
    return rows


def load_sampled_clip(path: Path, sample_fps: float, size: int, clip_length: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    frames: list[np.ndarray] = []
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        next_sample_t = 0.0
        frame_index = 0
        while True:
            ok, bgr = cap.read()
            if not ok or bgr is None:
                break
            source_time = frame_index / fps
            if source_time + 1e-9 >= next_sample_t:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                frames.append(cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR))
                next_sample_t += 1.0 / sample_fps
            frame_index += 1
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"no decodable frames: {path}")
    sampled = make_clip_tensor(np.stack(frames), clip_length)
    return sampled.permute(1, 2, 3, 0).contiguous().numpy()


def binary_metrics(rows: list[dict[str, object]], pred_key: str) -> dict[str, object]:
    tn = fp = fn = tp = 0
    for row in rows:
        y, pred = int(row["true_label"]), int(row[pred_key])
        if y == 0 and pred == 0: tn += 1
        elif y == 0 and pred == 1: fp += 1
        elif y == 1 and pred == 0: fn += 1
        else: tp += 1
    f1_0 = 2 * tn / max(1, 2 * tn + fp + fn)
    f1_1 = 2 * tp / max(1, 2 * tp + fp + fn)
    return {
        "n": len(rows), "correct": tn + tp, "accuracy": (tn + tp) / len(rows),
        "macro_f1": 0.5 * (f1_0 + f1_1), "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def exact_mcnemar(rows: list[dict[str, object]]) -> dict[str, object]:
    route_only = sum(int(r["route_on_pred"]) == int(r["true_label"]) and int(r["route_bypassed_pred"]) != int(r["true_label"]) for r in rows)
    bypass_only = sum(int(r["route_bypassed_pred"]) == int(r["true_label"]) and int(r["route_on_pred"]) != int(r["true_label"]) for r in rows)
    discordant = route_only + bypass_only
    if discordant == 0:
        p_value = 1.0
    else:
        lower = min(route_only, bypass_only)
        p_value = min(1.0, 2.0 * sum(math.comb(discordant, k) for k in range(lower + 1)) / (2 ** discordant))
    return {"route_on_only_correct": route_only, "route_bypassed_only_correct": bypass_only, "discordant": discordant, "exact_two_sided_p": p_value}


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
        route = binary_metrics(sample, "route_on_pred")["accuracy"]
        bypass = binary_metrics(sample, "route_bypassed_pred")["accuracy"]
        differences.append(float(route) - float(bypass))
    differences.sort()
    def quantile(q: float) -> float:
        index = (len(differences) - 1) * q
        lo, hi = math.floor(index), math.ceil(index)
        return differences[lo] if lo == hi else differences[lo] * (hi - index) + differences[hi] * (index - lo)
    return {
        "replicates": replicates, "seed": seed, "resampling_unit": "semantic_group_id",
        "group_count": len(keys), "accuracy_difference_route_on_minus_bypassed": {
            "mean": sum(differences) / len(differences), "ci95_percentile": [quantile(0.025), quantile(0.975)]
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--person-model", default="yolo11n.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--detector-device", default="0")
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--threshold", type=float, default=0.475)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = args.output_dir / "offline_policy_predictions.csv"
    completed: dict[str, dict[str, str]] = {}
    if args.resume and output_csv.exists():
        with output_csv.open(encoding="utf-8", newline="") as handle:
            completed = {row["video_id"]: row for row in csv.DictReader(handle)}
    elif output_csv.exists():
        raise FileExistsError(output_csv)

    manifest = read_manifest(args.manifest)
    device = torch.device(args.device)
    runtime = M3StreamingRuntime(
        RuntimeConfig(threshold=args.threshold, hash_candidates=True), device, "m3_route_bypassed",
        args.checkpoint, args.person_model, args.detector_device, args.tracker, False,
    )
    rows: list[dict[str, object]] = [dict(row) for row in completed.values()]
    write_header = not output_csv.exists()
    with output_csv.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        for index, item in enumerate(manifest, start=1):
            if item["video_id"] in completed:
                continue
            runtime.reset_stream_state()
            source = (args.repo_root / item["source_video"]).resolve()
            clip = load_sampled_clip(source, float(item["sample_fps"]), 224, 50)
            final: dict[str, object] | None = None
            for frame in clip:
                final = runtime.process(frame)
            assert final is not None
            calls = list(final.get("calls", []))
            candidates = list(final.get("candidates", []))
            selected = [call for call in calls if bool(call.get("route_decision"))]
            bypass_score = max((float(call["score"]) for call in calls), default=0.0)
            route_score = max((float(call["score"]) for call in selected), default=0.0)
            row: dict[str, object] = {
                "video_id": item["video_id"], "semantic_group_id": item["semantic_group_id"],
                "source_dataset": item["source_dataset"], "source_video": item["source_video"],
                "true_label": int(item["label"]),
                "route_on_pred": int(route_score >= args.threshold), "route_on_score": route_score,
                "route_bypassed_pred": int(bypass_score >= args.threshold), "route_bypassed_score": bypass_score,
                "eligible_candidates": len(candidates), "route_selected_candidates": len(selected),
                "candidate_ids": "|".join(str(c["candidate_id"]) for c in candidates),
                "candidate_sha256s": "|".join(str(c["tensor_sha256"]) for c in candidates),
            }
            writer.writerow(row)
            handle.flush()
            rows.append(row)
            print(f"[{index:03d}/526] {item['video_id']} eligible={len(candidates)} route={len(selected)}", flush=True)

    if len(rows) != 526:
        raise ValueError(f"incomplete output: {len(rows)}/526")
    by_id = {str(row["video_id"]): row for row in rows}
    ordered = [by_id[item["video_id"]] for item in manifest]
    eligible_count = sum(int(r["eligible_candidates"]) for r in ordered)
    selected_count = sum(int(r["route_selected_candidates"]) for r in ordered)
    eligible_video_count = sum(int(r["eligible_candidates"]) > 0 for r in ordered)
    selected_video_count = sum(int(r["route_selected_candidates"]) > 0 for r in ordered)
    analyzed_update_count = len(ordered) * 50
    payload = {
        "protocol": "jvcir_same_candidate_offline_policy_ablation_v11",
        "manifest_sha256": sha256(args.manifest), "checkpoint_sha256": sha256(args.checkpoint),
        "candidate_rule": "one common bypass-executed candidate set at each video's final T50 decision update; route-on retains calls with precomputed route_decision=true",
        "sampling": "decode at manifest sample_fps, resize RGB to 224x224, uniform/pad to T50",
        "route_on": binary_metrics(ordered, "route_on_pred"),
        "route_bypassed": binary_metrics(ordered, "route_bypassed_pred"),
        "candidate_counts": {
            "eligible": eligible_count,
            "route_selected": selected_count,
            "eligible_videos": eligible_video_count,
            "route_selected_videos": selected_video_count,
            "analyzed_updates": analyzed_update_count,
        },
        "invocation_rates": {
            "route_on_q_candidate": selected_count / max(1, eligible_count),
            "route_bypassed_q_candidate": 1.0 if eligible_count else 0.0,
            "route_on_q_video": selected_video_count / len(ordered),
            "route_bypassed_q_video": eligible_video_count / len(ordered),
            "route_on_q_update": selected_count / analyzed_update_count,
            "route_bypassed_q_update": eligible_count / analyzed_update_count,
            "q_update_denominator_rule": "classifier calls divided by all 50 sampled analyzed updates per held-out video",
        },
        "paired_mcnemar": exact_mcnemar(ordered),
        "grouped_bootstrap": grouped_bootstrap(ordered, args.bootstrap, 50900),
        "predictions_sha256": sha256(output_csv),
    }
    (args.output_dir / "offline_policy_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
