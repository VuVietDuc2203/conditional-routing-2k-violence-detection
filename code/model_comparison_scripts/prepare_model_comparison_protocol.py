#!/usr/bin/env python3
"""Freeze the nine-model comparison protocol without copying tensor caches."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


SEEDS = [50900, 50901, 50902]
EXPECTED_SPLITS = {"train": 2460, "val": 530, "test": 526}
MODEL_MATRIX: list[dict[str, Any]] = [
    {
        "model_id": "dense_wholeframe_movinet_a2",
        "paper_name": "Dense whole-frame MoViNet-A2",
        "runner": "movinet",
        "runner_variant": "M1",
        "clip_length": 50,
        "cache_profile": "wholeframe_rgb_t50_224",
        "source_manifest": "m1",
        "pretraining": "kinetics600_default",
    },
    {
        "model_id": "regularized_wholeframe_movinet_a2",
        "paper_name": "Regularized whole-frame MoViNet-A2",
        "runner": "movinet",
        "runner_variant": "M2",
        "clip_length": 50,
        "cache_profile": "wholeframe_rgb_t50_224",
        "source_manifest": "m1",
        "pretraining": "kinetics600_default",
    },
    {
        "model_id": "crowd_centric_movinet_a2",
        "paper_name": "Crowd-centric preprocessed MoViNet-A2",
        "runner": "movinet",
        "runner_variant": "M3",
        "clip_length": 50,
        "cache_profile": "movinet_preprocessed_t50_224",
        "source_manifest": "m3",
        "pretraining": "kinetics600_default",
    },
    {
        "model_id": "c3d",
        "paper_name": "C3D",
        "runner": "baseline",
        "runner_model": "c3d",
        "clip_length": 16,
        "cache_profile": "c3d_rgb_t16_112",
        "source_manifest": "m1",
        "pretraining": "scratch",
        "batch_size": 64,
    },
    {
        "model_id": "i3d",
        "paper_name": "I3D",
        "runner": "baseline",
        "runner_model": "i3d",
        "clip_length": 32,
        "cache_profile": "wholeframe_rgb_t32_224",
        "source_manifest": "m1",
        "pretraining": "kinetics400_default",
        "batch_size": 8,
    },
    {
        "model_id": "resnet_lstm",
        "paper_name": "ResNet-LSTM",
        "runner": "baseline",
        "runner_model": "resnet_lstm",
        "clip_length": 32,
        "cache_profile": "wholeframe_rgb_t32_224",
        "source_manifest": "m1",
        "pretraining": "imagenet1k_default",
        "batch_size": 16,
    },
    {
        "model_id": "slowfast",
        "paper_name": "SlowFast",
        "runner": "baseline",
        "runner_model": "slowfast",
        "clip_length": 32,
        "cache_profile": "slowfast_rgb_t32_224",
        "source_manifest": "m1",
        "pretraining": "kinetics400_default",
        "batch_size": 4,
    },
    {
        "model_id": "swin3d",
        "paper_name": "Swin3D",
        "runner": "baseline",
        "runner_model": "swin3d",
        "clip_length": 32,
        "cache_profile": "wholeframe_rgb_t32_224",
        "source_manifest": "m1",
        "pretraining": "kinetics400_default",
        "batch_size": 4,
    },
    {
        "model_id": "josenet",
        "paper_name": "JOSENet",
        "runner": "baseline",
        "runner_model": "josenet",
        "clip_length": 16,
        "cache_profile": "josenet_rgb_t16_224",
        "source_manifest": "m1",
        "pretraining": "scratch",
        "batch_size": 8,
    },
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_yaml(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def validate_source(frame: pd.DataFrame, name: str) -> None:
    required = {"video_id", "label", "split", "semantic_group_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")
    if len(frame) != sum(EXPECTED_SPLITS.values()):
        raise ValueError(f"{name} has {len(frame)} rows, expected {sum(EXPECTED_SPLITS.values())}")
    if frame["video_id"].astype(str).duplicated().any():
        raise ValueError(f"{name} contains duplicate video_id values")
    counts = frame["split"].astype(str).value_counts().to_dict()
    if counts != EXPECTED_SPLITS:
        raise ValueError(f"{name} split counts {counts}, expected {EXPECTED_SPLITS}")
    crossings = frame.groupby("semantic_group_id")["split"].nunique()
    if int((crossings > 1).sum()) != 0:
        raise ValueError(f"{name} contains semantic groups crossing splits")


def filtered_manifest(source: pd.DataFrame, profile: pd.DataFrame, profile_name: str) -> pd.DataFrame:
    if profile["video_id"].astype(str).duplicated().any():
        raise ValueError(f"Cache profile {profile_name} contains duplicate video_id values")
    source = source.copy()
    profile = profile.copy()
    source["video_id"] = source["video_id"].astype(str)
    profile["video_id"] = profile["video_id"].astype(str)
    cache_columns = [
        column
        for column in [
            "video_id",
            "cache_path",
            "label",
            "label_name",
            "clip_length",
            "height",
            "width",
            "dtype",
            "preprocess_type",
            "frame_count",
            "sample_fps",
            "sha256",
        ]
        if column in profile.columns
    ]
    joined = source.merge(
        profile[cache_columns],
        on="video_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "__cache"),
    )
    if joined["cache_path__cache" if "cache_path__cache" in joined.columns else "cache_path"].isna().any():
        missing = joined.loc[
            joined["cache_path__cache" if "cache_path__cache" in joined.columns else "cache_path"].isna(),
            "video_id",
        ].head(5).tolist()
        raise ValueError(f"Profile {profile_name} is missing frozen IDs, examples={missing}")
    if "label__cache" in joined.columns:
        mismatch = joined["label"].astype(int) != joined["label__cache"].astype(int)
        if mismatch.any():
            raise ValueError(f"Profile {profile_name} has {int(mismatch.sum())} label mismatches")
    for column in [
        "cache_path",
        "label_name",
        "clip_length",
        "height",
        "width",
        "dtype",
        "preprocess_type",
        "frame_count",
        "sample_fps",
        "sha256",
    ]:
        cache_column = f"{column}__cache"
        if cache_column in joined.columns:
            joined[column] = joined[cache_column]
            joined = joined.drop(columns=[cache_column])
    if "label__cache" in joined.columns:
        joined = joined.drop(columns=["label__cache"])
    validate_source(joined, f"filtered {profile_name}")
    return joined


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Freeze only the six non-MoViNet baselines; existing MoViNet artifacts remain reference controls.",
    )
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    output = args.output_root.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty protocol root: {output}")
    manifests_dir = output / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=False)

    split_root = repo / "pipeline_artifacts" / "stage4_prime_split_v2_20260716"
    cache_root = repo / "result" / "gpu_cache"
    sources = {
        "m1": pd.read_csv(split_root / "m1_stage4_prime_v2_manifest.csv"),
        "m3": pd.read_csv(split_root / "m3_stage4_prime_v2_manifest.csv"),
    }
    for name, frame in sources.items():
        validate_source(frame, name)
    if set(sources["m1"]["video_id"].astype(str)) != set(sources["m3"]["video_id"].astype(str)):
        raise ValueError("M1/M3 frozen source manifests do not contain identical IDs")

    profile_frames: dict[str, pd.DataFrame] = {}
    input_hashes: dict[str, str] = {}
    for source_name in ["m1", "m3"]:
        path = split_root / f"{source_name}_stage4_prime_v2_manifest.csv"
        input_hashes[str(path.relative_to(repo))] = sha256_file(path)

    selected_matrix = [spec for spec in MODEL_MATRIX if not args.baseline_only or spec["runner"] == "baseline"]
    registry: list[dict[str, Any]] = []
    for spec in selected_matrix:
        profile_name = str(spec["cache_profile"])
        profile_path = cache_root / profile_name / "manifest.csv"
        if profile_name not in profile_frames:
            if not profile_path.exists():
                raise FileNotFoundError(profile_path)
            profile_frames[profile_name] = pd.read_csv(profile_path)
            input_hashes[str(profile_path.relative_to(repo))] = sha256_file(profile_path)
        full = filtered_manifest(sources[str(spec["source_manifest"])], profile_frames[profile_name], profile_name)
        full_path = manifests_dir / f"{spec['model_id']}_full.csv"
        development_path = manifests_dir / f"{spec['model_id']}_development.csv"
        full.to_csv(full_path, index=False)
        development = full[full["split"].isin(["train", "val"])].copy()
        if len(development) != 2990 or development["split"].value_counts().to_dict() != {"train": 2460, "val": 530}:
            raise ValueError(f"Invalid development manifest for {spec['model_id']}")
        development.to_csv(development_path, index=False)
        item = dict(spec)
        item.update(
            {
                "full_manifest": str(full_path.relative_to(output)),
                "development_manifest": str(development_path.relative_to(output)),
                "full_manifest_sha256": sha256_file(full_path),
                "development_manifest_sha256": sha256_file(development_path),
                "full_rows": len(full),
                "development_rows": len(development),
            }
        )
        registry.append(item)

    paper_names = {
        "internal_to_paper": {
            "M1": "Dense whole-frame MoViNet-A2",
            "M2": "Regularized whole-frame MoViNet-A2",
            "M3_classifier": "Crowd-centric preprocessed MoViNet-A2",
            "M3_routed": "Kinematic-routed MoViNet-A2 system",
            "M1_stride50": "Sparse stride-50 whole-frame MoViNet-A2 control",
            "M3_gate_only": "Kinematic gate-only Stage 1",
        },
        "submission_facing_forbidden_tokens": ["M1", "M2", "M3"],
    }
    registry_path = output / "model_registry.yaml"
    names_path = output / "paper_name_map.yaml"
    write_json_yaml(registry_path, {"models": registry})
    write_json_yaml(names_path, paper_names)

    code_paths = [
        repo / "training_code" / "run_movinet_cached_experiments.py",
        repo / "training_code" / "run_jrtip_cached_experiments.py",
        repo / "data" / "processors" / "model_cache_adapters.py",
        Path(__file__).resolve(),
    ]
    for path in code_paths:
        input_hashes[str(path.relative_to(repo))] = sha256_file(path)

    freeze = {
        "protocol_version": "model_comparison_amendment_v2_20260717",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": {"total": 3516, "splits": EXPECTED_SPLITS, "semantic_group_crossings": 0},
        "models": len(registry),
        "training_runs": len(registry) * len(SEEDS),
        "existing_reference_controls": [
            "Dense whole-frame MoViNet-A2",
            "Regularized whole-frame MoViNet-A2",
            "Crowd-centric preprocessed MoViNet-A2",
        ] if args.baseline_only else [],
        "training_scope": "six non-MoViNet baselines only" if args.baseline_only else "nine-model matrix",
        "seeds": SEEDS,
        "checkpoint_selection": {
            "data": "validation_only",
            "objective": "0.5 * balanced_accuracy + 0.5 * macro_f1",
            "epochs": 30,
            "early_stopping_patience": 6,
        },
        "batch_policy": {
            "mode": "fixed_per_architecture",
            "reason": "avoid WDDM shared-memory oversubscription and near-OOM auto-batch probes",
            "auto_batch_disabled": True,
        },
        "threshold_selection": {
            "data": "validation_only",
            "grid_start": 0.35,
            "grid_stop": 0.65,
            "grid_step": 0.005,
            "primary": "macro_f1",
            "tie_breakers": ["balanced_accuracy", "distance_to_0.5"],
        },
        "test_policy": "No test rows are exposed before all checkpoints and thresholds are frozen.",
        "statistical_policy": {
            "bootstrap_replicates": 10000,
            "resampling_unit": "semantic_group_id",
            "paired_test": "exact McNemar",
        },
        "input_sha256": input_hashes,
        "model_registry_sha256": sha256_file(registry_path),
        "paper_name_map_sha256": sha256_file(names_path),
    }
    freeze_path = output / "protocol_freeze.yaml"
    write_json_yaml(freeze_path, freeze)
    inventory = []
    for path in sorted(output.rglob("*")):
        if path.is_file():
            inventory.append(
                {
                    "path": str(path.relative_to(output)).replace("\\", "/"),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    (output / "protocol_inventory.json").write_text(
        json.dumps({"files": inventory}, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(output),
                "models": len(registry),
                "runs": len(registry) * len(SEEDS),
                "baseline_only": bool(args.baseline_only),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
