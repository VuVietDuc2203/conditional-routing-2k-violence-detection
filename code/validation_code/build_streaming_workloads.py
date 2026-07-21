"""Build reproducible, one-camera logical replay streams from held-out clips.

This is an offline dataset-preparation utility.  It may read labels to compose
controlled workloads, but it writes labels only to sidecar manifests; the
streaming runtime never receives those files.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def duration_seconds(path: Path, ffprobe: str) -> float:
    output = subprocess.check_output(
        [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
        text=True,
    ).strip()
    return max(0.01, float(output))


def source_path(row: pd.Series) -> Path:
    raw = Path(str(row["source_video"]))
    return raw if raw.is_absolute() else REPO_ROOT / raw


def choose_rows(pool: pd.DataFrame, target_seconds: float, rng: random.Random, ffprobe: str) -> list[dict]:
    rows = pool.sample(frac=1, random_state=rng.randrange(2**31)).to_dict("records")
    chosen: list[dict] = []
    total = 0.0
    index = 0
    while total < target_seconds:
        row = rows[index % len(rows)]
        path = source_path(pd.Series(row))
        if path.exists():
            item = dict(row)
            item["resolved_source"] = str(path)
            item["duration_sec"] = duration_seconds(path, ffprobe)
            chosen.append(item)
            total += item["duration_sec"]
        index += 1
        if index > len(rows) * 20:
            raise RuntimeError("Could not assemble requested duration from available clips")
    return chosen


def mixed_rows(normal: pd.DataFrame, violent: pd.DataFrame, target_seconds: float, violent_ratio: float,
               rng: random.Random, ffprobe: str) -> list[dict]:
    target_violent = target_seconds * violent_ratio
    violent_rows = choose_rows(violent, target_violent, rng, ffprobe)
    normal_rows = choose_rows(normal, max(0.0, target_seconds - sum(x["duration_sec"] for x in violent_rows)), rng, ffprobe)
    combined = normal_rows + violent_rows
    rng.shuffle(combined)
    return combined


def normalize_and_concat(rows: list[dict], output: Path, work_dir: Path, ffmpeg: str,
                         ffprobe: str, width: int, height: int, fps: float) -> None:
    segment_dir = work_dir / "segments"
    segment_dir.mkdir(parents=True, exist_ok=True)
    normalized: list[Path] = []
    for index, row in enumerate(rows):
        segment = segment_dir / f"{index:04d}.mp4"
        command = [
            ffmpeg, "-y", "-i", row["resolved_source"], "-an",
            "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,fps={fps}",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", str(segment),
        ]
        subprocess.run(command, check=True)
        normalized.append(segment)
    # The encoded segment durations, rather than source metadata durations,
    # define the replay timeline. These labels stay in an offline sidecar and
    # are never read by the runtime benchmark.
    cursor = 0.0
    for row, segment in zip(rows, normalized):
        encoded_duration = duration_seconds(segment, ffprobe)
        row["stream_start_sec"] = cursor
        row["stream_end_sec"] = cursor + encoded_duration
        row["encoded_duration_sec"] = encoded_duration
        row["event_type"] = "violent_clip" if int(row["label"]) == 1 else "normal_clip"
        cursor += encoded_duration
    list_file = work_dir / "concat.txt"
    list_file.write_text(
        "".join(f"file '{path.resolve().as_posix()}'\n" for path in normalized),
        encoding="utf-8",
    )
    subprocess.run(
        [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(list_file.resolve()), "-c", "copy", str(output.resolve())],
        check=True,
    )


def build(name: str, rows: list[dict], args: argparse.Namespace) -> None:
    output = args.output_dir / f"{name}_1440p30.mp4"
    work_dir = args.output_dir / name
    work_dir.mkdir(parents=True, exist_ok=False)
    for item in rows:
        item["runtime_label_access"] = "forbidden"
    if not args.manifest_only:
        normalize_and_concat(rows, output, work_dir, args.ffmpeg, args.ffprobe, args.width, args.height, args.fps)
    else:
        # Manifest-only workload selection deliberately has no replay timeline.
        # A downstream benchmark must not mistake source metadata for event time.
        for item in rows:
            item["event_type"] = "offline_label_only"
    (work_dir / "sidecar_events.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (work_dir / "selection.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("result/gpu_cache/wholeframe_rgb_t50_224/manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("result/streaming_2k/workloads"))
    parser.add_argument("--duration-sec", type=float, default=600.0)
    parser.add_argument("--mixed-violence-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=50900)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--width", type=int, default=2560)
    parser.add_argument("--height", type=int, default=1440)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest)
    test = manifest[manifest["split"] == "test"].reset_index(drop=True)
    normal, violent = test[test["label"] == 0], test[test["label"] == 1]
    if normal.empty or violent.empty:
        raise RuntimeError("Held-out split must contain both classes")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rng = random.Random(args.seed)
    build("normal_only", choose_rows(normal, args.duration_sec, rng, args.ffprobe), args)
    build("mixed_controlled", mixed_rows(normal, violent, args.duration_sec, args.mixed_violence_ratio, rng, args.ffprobe), args)
    build("kinetic_rich", choose_rows(violent, args.duration_sec, rng, args.ffprobe), args)
    (args.output_dir / "workload_manifest.json").write_text(
        json.dumps({"seed": args.seed, "duration_sec": args.duration_sec, "resolution": [args.width, args.height],
                    "source_fps": args.fps, "runtime_label_access": "forbidden"}, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
