"""Run all non-MoViNet JRTIP baseline/SOTA jobs from result/gpu_cache."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_MODULE = "training_code.run_jrtip_cached_experiments"
AVAILABLE_MODELS = ["c3d", "i3d", "resnet_lstm", "slowfast", "swin3d", "josenet"]
UNAVAILABLE_OFFICIAL_SOTA: list[str] = []
DEFAULT_MODELS = AVAILABLE_MODELS + UNAVAILABLE_OFFICIAL_SOTA
ALLOWED_CLIP_LENGTHS = {
    "c3d": {16},
    "i3d": {16, 32, 64},
    "resnet_lstm": {16, 32, 64},
    "slowfast": {32, 64},
    "swin3d": {16, 32, 64},
    "josenet": {16},
}
PREFERRED_CLIP_LENGTHS = {
    "c3d": 16,
    "i3d": 32,
    "resnet_lstm": 32,
    "slowfast": 32,
    "swin3d": 32,
    "josenet": 16,
}
SCOPED_PRETRAINED_MODELS = {"i3d", "resnet_lstm", "slowfast", "swin3d"}
PRETRAINED_UNAVAILABLE = {
    "c3d": "true C3D pretrained weights are not configured",
    "josenet": "JOSENet pretrained/self-supervised checkpoints are not configured",
}
DEFAULT_BATCH_BY_MODEL = {
    "swin3d": 8,
    "josenet": 8,
}
LEGACY_MARKERS = (
    "Ver1InferenceDataset",
    "extract_ver1",
    "results/ver1_inference",
    "data/preprocessed",
    "datasets/preprocessed",
)


def reject_legacy_path(path: str | Path, field_name: str) -> None:
    normalized = str(path).replace("\\", "/")
    for marker in LEGACY_MARKERS:
        if marker in normalized:
            raise ValueError(f"{field_name} points to legacy cache/data: {path}")


def require_existing_path(path: str | Path, field_name: str) -> Path:
    resolved = Path(path)
    reject_legacy_path(resolved, field_name)
    if not resolved.exists():
        raise FileNotFoundError(f"{field_name} not found: {path}")
    return resolved


def require_result_output(path: str | Path) -> Path:
    output_root = Path(path)
    parts = [part.lower() for part in output_root.parts]
    if "result" not in parts and (not parts or parts[0] != "result"):
        raise ValueError(f"output-root must be under result/: {path}")
    return output_root


def parse_models(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return AVAILABLE_MODELS.copy()
    models = [item.strip().lower().replace("-", "_") for item in raw.split(",") if item.strip()]
    unknown = sorted(set(models) - set(DEFAULT_MODELS))
    if unknown:
        raise ValueError(f"Unsupported models: {unknown}. Supported: {DEFAULT_MODELS}")
    return models


def load_metrics(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_command(args: argparse.Namespace, model: str, clip_length: int, pretrained: bool) -> list[str]:
    batch_size = args.batch_size
    if batch_size is None:
        batch_size = DEFAULT_BATCH_BY_MODEL.get(model, 4)
        if int(clip_length) >= 64:
            batch_size = max(1, batch_size // 4)
        elif int(clip_length) >= 32:
            batch_size = max(1, batch_size // 2)
    cmd = [
        sys.executable,
        "-m",
        RUNNER_MODULE,
        "--model",
        model,
        "--clip-length",
        str(clip_length),
        "--cache-root",
        str(args.cache_root),
        "--output-root",
        str(args.output_root),
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(0 if model == "josenet" else args.num_workers),
        "--vram-target",
        str(args.vram_target),
        "--max-auto-batch",
        str(args.max_auto_batch),
    ]
    if args.lr is not None:
        cmd.extend(["--lr", str(args.lr)])
    if args.weight_decay is not None:
        cmd.extend(["--weight-decay", str(args.weight_decay)])
    if args.device:
        cmd.extend(["--device", args.device])
    cmd.extend(["--seed", str(args.seed)])
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    if pretrained:
        cmd.append("--pretrained")
    if args.amp:
        cmd.append("--amp")
    if args.freeze_backbone:
        cmd.append("--freeze-backbone")
    if args.smoke:
        cmd.append("--smoke")
    return cmd


def is_valid_clip_length(model: str, clip_length: int) -> tuple[bool, str]:
    allowed = ALLOWED_CLIP_LENGTHS.get(model)
    if allowed is None or int(clip_length) in allowed:
        return True, ""
    allowed_text = ",".join(str(x) for x in sorted(allowed))
    return False, f"{model} supports only clip lengths: {allowed_text}"


def is_pretrained_available(model: str) -> tuple[bool, str]:
    reason = PRETRAINED_UNAVAILABLE.get(model.lower().replace("-", "_"))
    if reason:
        return False, reason
    return True, ""


def should_use_pretrained(args: argparse.Namespace, model: str) -> bool:
    if args.pretrained_policy == "none":
        return False
    if args.pretrained_policy == "all":
        return True
    return model.lower().replace("-", "_") in SCOPED_PRETRAINED_MODELS


def is_in_requested_scope(args: argparse.Namespace, model: str, clip_length: int) -> tuple[bool, str]:
    if not args.single_best_baseline_scope:
        return True, ""
    preferred = PREFERRED_CLIP_LENGTHS.get(model.lower().replace("-", "_"))
    if preferred is None or int(clip_length) == int(preferred):
        return True, ""
    return False, f"{model} is scoped to clip_length={preferred} for this cached benchmark plan"


def write_summary(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "clip_length",
        "status",
        "returncode",
        "elapsed_seconds",
        "output_dir",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "f1_macro",
        "tn",
        "fp",
        "fn",
        "tp",
        "peak_vram_mb",
        "cache_clips_per_sec",
        "cache_mean_read_ms",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> list[dict]:
    args.cache_root = require_existing_path(args.cache_root, "cache_root")
    args.output_root = require_result_output(args.output_root)
    models = parse_models(args.models)
    rows: list[dict] = []

    for model in models:
        for clip_length in args.clip_lengths:
            run_dir = args.output_root / model / f"t{clip_length}"
            metrics_path = run_dir / "metrics.json"
            valid, reason = is_valid_clip_length(model, int(clip_length))
            if not valid:
                rows.append(
                    {
                        "model": model,
                        "clip_length": clip_length,
                        "status": "skipped_invalid_clip_length",
                        "returncode": 0,
                        "elapsed_seconds": 0.0,
                        "output_dir": str(run_dir),
                        "accuracy": "",
                        "precision": "",
                        "recall": "",
                        "f1": "",
                        "tn": "",
                        "fp": "",
                        "fn": "",
                        "tp": "",
                        "error": reason,
                    }
                )
                write_summary(args.summary_csv, rows)
                continue
            in_scope, scope_reason = is_in_requested_scope(args, model, int(clip_length))
            if not in_scope:
                rows.append(
                    {
                        "model": model,
                        "clip_length": clip_length,
                        "status": "skipped_not_in_requested_scope",
                        "returncode": 0,
                        "elapsed_seconds": 0.0,
                        "output_dir": str(run_dir),
                        "accuracy": "",
                        "precision": "",
                        "recall": "",
                        "f1": "",
                        "tn": "",
                        "fp": "",
                        "fn": "",
                        "tp": "",
                        "error": scope_reason,
                    }
                )
                write_summary(args.summary_csv, rows)
                continue
            use_pretrained = should_use_pretrained(args, model)
            pretrained_ok, pretrained_reason = is_pretrained_available(model)
            if use_pretrained and not pretrained_ok:
                rows.append(
                    {
                        "model": model,
                        "clip_length": clip_length,
                        "status": "skipped_pretrained_unavailable",
                        "returncode": 0,
                        "elapsed_seconds": 0.0,
                        "output_dir": str(run_dir),
                        "accuracy": "",
                        "precision": "",
                        "recall": "",
                        "f1": "",
                        "tn": "",
                        "fp": "",
                        "fn": "",
                        "tp": "",
                        "error": pretrained_reason,
                    }
                )
                write_summary(args.summary_csv, rows)
                continue
            if args.skip_existing and metrics_path.exists():
                metrics = load_metrics(metrics_path)
                test = metrics.get("test", {})
                rows.append(
                    {
                        "model": model,
                        "clip_length": clip_length,
                        "status": "skipped_existing",
                        "returncode": 0,
                        "elapsed_seconds": 0.0,
                        "output_dir": str(run_dir),
                        "accuracy": test.get("accuracy"),
                        "balanced_accuracy": test.get("balanced_accuracy"),
                        "precision": test.get("precision"),
                        "recall": test.get("recall"),
                        "f1": test.get("f1"),
                        "f1_macro": test.get("f1_macro"),
                        "tn": test.get("tn"),
                        "fp": test.get("fp"),
                        "fn": test.get("fn"),
                        "tp": test.get("tp"),
                        "peak_vram_mb": metrics.get("peak_vram_mb", ""),
                        "cache_clips_per_sec": metrics.get("cache_read_throughput", {}).get("clips_per_sec", ""),
                        "cache_mean_read_ms": metrics.get("cache_read_throughput", {}).get("mean_read_ms_per_clip", ""),
                        "error": "",
                    }
                )
                continue

            cmd = build_command(args, model, int(clip_length), use_pretrained)
            print("RUN", " ".join(cmd), flush=True)
            if args.dry_run:
                rows.append(
                    {
                        "model": model,
                        "clip_length": clip_length,
                        "status": "dry_run",
                        "returncode": 0,
                        "elapsed_seconds": 0.0,
                        "output_dir": str(run_dir),
                        "accuracy": "",
                        "precision": "",
                        "recall": "",
                        "f1": "",
                        "tn": "",
                        "fp": "",
                        "fn": "",
                        "tp": "",
                        "error": "",
                    }
                )
                continue

            start = time.time()
            proc = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
            elapsed = time.time() - start
            metrics = load_metrics(metrics_path)
            test = metrics.get("test", {})
            status = "ok" if proc.returncode == 0 else "failed"
            row = {
                "model": model,
                "clip_length": clip_length,
                "status": status,
                "returncode": proc.returncode,
                "elapsed_seconds": elapsed,
                "output_dir": str(run_dir),
                "accuracy": test.get("accuracy", ""),
                "balanced_accuracy": test.get("balanced_accuracy", ""),
                "precision": test.get("precision", ""),
                "recall": test.get("recall", ""),
                "f1": test.get("f1", ""),
                "f1_macro": test.get("f1_macro", ""),
                "tn": test.get("tn", ""),
                "fp": test.get("fp", ""),
                "fn": test.get("fn", ""),
                "tp": test.get("tp", ""),
                "peak_vram_mb": metrics.get("peak_vram_mb", ""),
                "cache_clips_per_sec": metrics.get("cache_read_throughput", {}).get("clips_per_sec", ""),
                "cache_mean_read_ms": metrics.get("cache_read_throughput", {}).get("mean_read_ms_per_clip", ""),
                "error": "" if proc.returncode == 0 else f"runner returned {proc.returncode}",
            }
            rows.append(row)
            write_summary(args.summary_csv, rows)
            if proc.returncode != 0 and not args.continue_on_error:
                raise RuntimeError(f"Job failed: model={model} clip_length={clip_length}")

    write_summary(args.summary_csv, rows)
    return rows


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all non-MoViNet JRTIP baselines from result/gpu_cache.")
    parser.add_argument(
        "--models",
        default="all",
        help=(
            "Comma list or 'all'. 'all' includes trainable cached-video models "
            "(c3d,i3d,resnet_lstm,slowfast,swin3d,josenet). VideoMamba is excluded."
        ),
    )
    parser.add_argument("--clip-lengths", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--cache-root", type=Path, default=Path("result/gpu_cache"))
    parser.add_argument("--output-root", type=Path, default=Path("result/cached_experiments"))
    parser.add_argument("--summary-csv", type=Path, default=Path("result/cached_experiments/summary.csv"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--pretrained", action="store_true", help="Shortcut for --pretrained-policy all.")
    parser.add_argument("--pretrained-policy", choices=["none", "scoped", "all"], default="scoped")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--vram-target", type=float, default=0.92)
    parser.add_argument("--max-auto-batch", type=int, default=128)
    parser.add_argument("--single-best-baseline-scope", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args(argv)
    if args.pretrained:
        args.pretrained_policy = "all"
    return args


def main() -> None:
    args = parse_args()
    rows = run(args)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
