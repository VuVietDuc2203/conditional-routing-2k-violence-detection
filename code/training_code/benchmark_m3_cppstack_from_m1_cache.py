
"""
Benchmark MoViNet M3 as a full cpp-stack gate pipeline over M1 cached clips.

Input is the M1 whole-frame cache (default: result/gpu_cache/wholeframe_rgb_t50_224).
For each cached 50-frame clip, the script runs YOLOv11/ByteTrack, crowd
clustering, kinematic gate, and cpp-stack crop. MoViNet M3 is called only when
the pipeline produces a complete 50-frame stack; otherwise the prediction is
normal without model inference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.scripts.build_movinet_preprocess_cache import (  # noqa: E402
    bbox_iou_xyxy,
    build_movinet_sequence_cpp_like,
    center_xyxy,
    clamp_xyxy,
    cluster_person_indices,
    get_person_model,
    make_clip_tensor,
    match_crowd_track,
    union_xyxy,
)
from data.scripts.build_movinet_preprocess_cache import hdbscan as HDBSCAN_BACKEND  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
from training_code.run_movinet_cached_experiments import create_model  # noqa: E402


SUMMARY_FIELDS = [
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
    "num_samples",
    "m3_calls",
    "skipped_normal",
    "pipeline_success_rate",
    "fps_videos_per_sec",
    "latency_mean_ms_per_video",
    "yolo_track_ms",
    "yolo_inference_only_ms",
    "bytetrack_ms",
    "hdbscan_ms",
    "kinetic_gate_ms",
    "cpp_crop_ms",
    "m3_inference_ms",
    "peak_vram_mb",
]

PER_VIDEO_FIELDS = [
    "video_id",
    "true_label",
    "pred_label",
    "score_violence",
    "preprocess_status",
    "skip_reason",
    "pipeline_mode",
    "detector_mode",
    "person_frames",
    "person_kinetic_frames",
    "crowd_frames",
    "gate_frames",
    "completed_stacks",
    "total_ms",
    "yolo_track_ms",
    "yolo_inference_only_ms",
    "bytetrack_ms",
    "hdbscan_ms",
    "gate_ms",
    "cpp_crop_ms",
    "m3_ms",
    "cache_path",
    "source_video",
]


def device_from_arg(raw: str) -> torch.device:
    if raw == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(raw)


def cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_state(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    elif "model" in checkpoint:
        state = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    else:
        raise RuntimeError(f"No model state found in checkpoint: {checkpoint_path}")
    model.load_state_dict(state)


def resolve_cache_path(cache_root: Path, profile_dir: Path, cache_path: str) -> Path:
    path = Path(cache_path)
    if path.is_absolute():
        return path
    root_candidate = cache_root / path
    if root_candidate.exists():
        return root_candidate
    return profile_dir / path


def load_cached_rgb_clip(path: Path, clip_length: int) -> np.ndarray:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    video = payload["video"]
    if not torch.is_tensor(video):
        video = torch.as_tensor(video)
    if video.dtype != torch.uint8:
        video = video.to(torch.uint8)
    if tuple(video.shape) != (3, int(clip_length), 224, 224):
        raise ValueError(f"Expected cached video shape (3,{clip_length},224,224), got {tuple(video.shape)} at {path}")
    return video.permute(1, 2, 3, 0).contiguous().numpy()


def reset_yolo_tracker_for_clip(model: Any) -> None:
    predictor = getattr(model, "predictor", None)
    trackers = getattr(predictor, "trackers", None)
    if trackers:
        for tracker in trackers:
            reset = getattr(tracker, "reset", None)
            if callable(reset):
                reset()


def result_speed_ms(results: list[Any]) -> dict[str, float]:
    preprocess = 0.0
    inference = 0.0
    postprocess = 0.0
    for result in results:
        speed = getattr(result, "speed", {}) or {}
        preprocess += float(speed.get("preprocess", 0.0))
        inference += float(speed.get("inference", 0.0))
        postprocess += float(speed.get("postprocess", 0.0))
    return {
        "preprocess_ms": preprocess,
        "inference_ms": inference,
        "postprocess_ms": postprocess,
    }


def run_yolo_batch(frames_rgb: np.ndarray, args: argparse.Namespace) -> tuple[list[Any], float, float]:
    model = get_person_model(str(args.person_model))
    reset_yolo_tracker_for_clip(model)
    frames_bgr = [cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR) for frame_rgb in frames_rgb]
    start = time.perf_counter()
    results = model.predict(
        source=frames_bgr,
        batch=int(args.yolo_batch_size) if int(args.yolo_batch_size) > 0 else len(frames_bgr),
        conf=float(args.person_conf),
        classes=[0],
        device=str(args.detector_device),
        verbose=False,
        imgsz=int(args.detector_imgsz),
        half=bool(args.half),
        stream=False,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    results_list = list(results)
    speeds = result_speed_ms(results_list)
    return results_list, elapsed_ms, float(speeds["inference_ms"])


def run_yolo_track_per_frame(frames_rgb: np.ndarray, args: argparse.Namespace) -> tuple[list[Any], float, float, float]:
    model = get_person_model(str(args.person_model))
    reset_yolo_tracker_for_clip(model)
    results: list[Any] = []
    total_ms = 0.0
    inference_only_ms = 0.0
    bytetrack_ms = 0.0
    first_frame = True
    for frame_rgb in frames_rgb:
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        start = time.perf_counter()
        result = model.track(
            source=frame_bgr,
            persist=not first_frame,
            tracker=str(args.tracker),
            conf=float(args.person_conf),
            classes=[0],
            device=str(args.detector_device),
            verbose=False,
            imgsz=int(args.detector_imgsz),
            half=bool(args.half),
            stream=False,
        )[0]
        first_frame = False
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        speed = getattr(result, "speed", {}) or {}
        detector_ms = float(speed.get("preprocess", 0.0) + speed.get("inference", 0.0) + speed.get("postprocess", 0.0))
        inference_only_ms += float(speed.get("inference", 0.0))
        total_ms += elapsed_ms
        bytetrack_ms += max(0.0, elapsed_ms - detector_ms) if detector_ms > 0 else 0.0
        results.append(result)
    return results, total_ms, inference_only_ms, bytetrack_ms


def run_cppstack_on_cached_frames(frames_rgb: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:

    prev_crowd_tracks: list[dict[str, Any]] = []
    lost_crowd_tracks: list[dict[str, Any]] = []
    prev_person_tracks: list[dict[str, Any]] = []
    movinet_stacks: dict[int, dict[str, Any]] = {}
    consecutive_kinetic_count: dict[int, int] = {}
    consecutive_person_kinetic_count: dict[int, int] = {}
    next_crowd_track_id = 1
    next_person_track_id = 1

    person_frames = 0
    crowd_frames = 0
    gate_frames = 0
    person_kinetic_frames = 0
    person_kinetic_pass = False
    completed_stacks = 0
    completed_sequence: np.ndarray | None = None
    full_clip_snapshots: list[dict[str, Any]] = []
    last_best_crowd_box: np.ndarray | None = None

    yolo_track_ms: list[float] = []
    yolo_inference_only_ms: list[float] = []
    bytetrack_ms: list[float] = []
    hdbscan_ms: list[float] = []
    gate_ms: list[float] = []
    cpp_crop_ms = 0.0

    frame_h, frame_w = frames_rgb.shape[1:3]
    if bool(args.yolo_batch):
        yolo_results, yolo_total_ms, yolo_inference_total_ms = run_yolo_batch(frames_rgb, args)
        yolo_per_frame_ms = yolo_total_ms / max(1, len(yolo_results))
        yolo_inference_per_frame_ms = yolo_inference_total_ms / max(1, len(yolo_results))
        bytetrack_total_ms = 0.0
        detector_mode = "batch_predict"
    else:
        yolo_results, yolo_total_ms, yolo_inference_total_ms, bytetrack_total_ms = run_yolo_track_per_frame(frames_rgb, args)
        yolo_per_frame_ms = yolo_total_ms / max(1, len(yolo_results))
        yolo_inference_per_frame_ms = yolo_inference_total_ms / max(1, len(yolo_results))
        detector_mode = "track_per_frame"

    for frame_rgb, result in zip(frames_rgb, yolo_results):
        yolo_track_ms.append(yolo_per_frame_ms)
        yolo_inference_only_ms.append(yolo_inference_per_frame_ms)
        bytetrack_ms.append(bytetrack_total_ms / max(1, len(yolo_results)) if not bool(args.yolo_batch) else 0.0)

        if result.boxes is None or result.boxes.xyxy is None or len(result.boxes) == 0:
            boxes = np.empty((0, 4), dtype=np.float32)
            person_ids: list[int] = []
        else:
            boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
            if result.boxes.id is not None:
                person_ids = result.boxes.id.detach().cpu().numpy().astype(int).tolist()
            else:
                person_ids = [-1] * len(boxes)

        previous_person_tracks = prev_person_tracks
        updated_person_tracks: list[dict[str, Any]] = []
        used_prev_person = [False] * len(previous_person_tracks)
        for box_idx, box in enumerate(boxes):
            curr_person_id = int(person_ids[box_idx]) if box_idx < len(person_ids) else -1
            best_idx = -1
            best_score = -1.0
            for prev_idx, prev_track in enumerate(previous_person_tracks):
                if used_prev_person[prev_idx]:
                    continue
                prev_person_id = int(prev_track.get("person_id", -1))
                if curr_person_id >= 0 and prev_person_id == curr_person_id:
                    score = 2.0
                else:
                    iou = bbox_iou_xyxy(prev_track["bbox"], box)
                    dist = float(np.linalg.norm(center_xyxy(prev_track["bbox"]) - center_xyxy(box)))
                    diag = max(1.0, float(np.linalg.norm([box[2] - box[0], box[3] - box[1]])))
                    score = iou if dist / diag <= float(args.person_match_dist) else -1.0
                if score > best_score:
                    best_score = score
                    best_idx = prev_idx

            if best_idx >= 0 and best_score > 0:
                prev_track = previous_person_tracks[best_idx]
                used_prev_person[best_idx] = True
                track_id = int(prev_track["track_id"])
                iou = bbox_iou_xyxy(prev_track["bbox"], box)
                diag = max(1.0, float(np.linalg.norm([box[2] - box[0], box[3] - box[1]])))
                velocity_norm = float(np.linalg.norm(center_xyxy(box) - center_xyxy(prev_track["bbox"])) / diag)
                kinetic = iou <= float(args.iou_gate) or velocity_norm >= float(args.velocity_gate)
                if kinetic:
                    consecutive_person_kinetic_count[track_id] = consecutive_person_kinetic_count.get(track_id, 0) + 1
                else:
                    consecutive_person_kinetic_count[track_id] = 0
                if consecutive_person_kinetic_count[track_id] >= int(args.kappa_frames):
                    person_kinetic_pass = True
                    person_kinetic_frames += 1
            else:
                track_id = next_person_track_id
                next_person_track_id += 1
                consecutive_person_kinetic_count[track_id] = 0

            updated_person_tracks.append(
                {
                    "track_id": int(track_id),
                    "person_id": int(curr_person_id),
                    "bbox": box.copy(),
                }
            )
        prev_person_tracks = updated_person_tracks

        t0 = time.perf_counter()
        crowd_boxes: list[np.ndarray] = []
        crowd_person_ids_list: list[list[int]] = []
        if len(boxes) > 0:
            person_frames += 1
            for idxs in cluster_person_indices(
                boxes,
                int(args.cluster_min_pts),
                float(args.hdbscan_epsilon),
            ):
                crowd_boxes.append(union_xyxy(boxes[idxs]))
                crowd_person_ids_list.append([int(person_ids[i]) for i in idxs if int(person_ids[i]) >= 0])
            if not crowd_boxes and int(args.crowd_fallback_min_persons) > 0 and len(boxes) >= int(args.crowd_fallback_min_persons):
                idxs = list(range(len(boxes)))
                crowd_boxes.append(union_xyxy(boxes))
                crowd_person_ids_list.append([int(person_ids[i]) for i in idxs if int(person_ids[i]) >= 0])
        hdbscan_ms.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        group_track_ids: list[int] = []
        previous_track_by_id = {int(t["track_id"]): t for t in prev_crowd_tracks}
        best_frame_crowd_box: np.ndarray | None = None

        if crowd_boxes:
            crowd_frames += 1
            best_frame_crowd_box = max(
                crowd_boxes,
                key=lambda box: max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1])),
            )
            last_best_crowd_box = best_frame_crowd_box.copy()
            available_tracks: list[dict[str, Any]] = [{**prev, "is_lost": False} for prev in prev_crowd_tracks]
            for lost in lost_crowd_tracks:
                if int(lost["frames_since_lost"]) <= int(args.crowd_retain_frames):
                    available_tracks.append(
                        {
                            "track_id": lost["track_id"],
                            "bbox": lost["last_bbox"],
                            "person_ids": lost["last_person_ids"],
                            "is_lost": True,
                        }
                    )

            track_used = [False] * len(available_tracks)
            assignment: dict[int, int] = {}
            crowd_indices = sorted(
                range(len(crowd_boxes)),
                key=lambda idx: len(crowd_person_ids_list[idx]),
                reverse=True,
            )
            for g_idx in crowd_indices:
                best_idx, best_score = match_crowd_track(
                    crowd_boxes[g_idx],
                    crowd_person_ids_list[g_idx],
                    available_tracks,
                    track_used,
                    use_fallback_thresholds=False,
                )
                if best_idx < 0 or best_score < 0:
                    best_idx, best_score = match_crowd_track(
                        crowd_boxes[g_idx],
                        crowd_person_ids_list[g_idx],
                        available_tracks,
                        track_used,
                        use_fallback_thresholds=True,
                    )
                if best_idx >= 0 and best_score > 0:
                    assignment[g_idx] = int(available_tracks[best_idx]["track_id"])
                    track_used[best_idx] = True
                else:
                    assignment[g_idx] = next_crowd_track_id
                    next_crowd_track_id += 1

            group_track_ids = [assignment[idx] for idx in range(len(crowd_boxes))]
            matched_ids = set(group_track_ids)
            lost_crowd_tracks = [lost for lost in lost_crowd_tracks if int(lost["track_id"]) not in matched_ids]
            prev_crowd_tracks = [
                {
                    "track_id": int(tid),
                    "bbox": crowd_boxes[idx],
                    "person_ids": crowd_person_ids_list[idx],
                }
                for idx, tid in enumerate(group_track_ids)
            ]
        else:
            for prev in prev_crowd_tracks:
                if not any(int(lost["track_id"]) == int(prev["track_id"]) for lost in lost_crowd_tracks):
                    lost_crowd_tracks.append(
                        {
                            "track_id": int(prev["track_id"]),
                            "last_bbox": prev["bbox"],
                            "last_person_ids": prev["person_ids"],
                            "frames_since_lost": 0,
                        }
                    )

        for lost in lost_crowd_tracks:
            lost["frames_since_lost"] = int(lost["frames_since_lost"]) + 1
        lost_crowd_tracks = [
            lost for lost in lost_crowd_tracks if int(lost["frames_since_lost"]) <= int(args.crowd_retain_frames)
        ]

        full_clip_snapshots.append(
            {
                "frame": frame_rgb.copy(),
                "crowd_box": (
                    best_frame_crowd_box.copy()
                    if best_frame_crowd_box is not None
                    else (last_best_crowd_box.copy() if last_best_crowd_box is not None else np.zeros(4, dtype=np.float32))
                ),
                "has_crowd": best_frame_crowd_box is not None,
            }
        )

        for g_idx, track_id in enumerate(group_track_ids):
            crowd_box = crowd_boxes[g_idx]
            if bool(args.kinetic_gate):
                should_start_stack = bool(person_kinetic_pass)
            else:
                should_start_stack = True

            already_stacking = bool(movinet_stacks.get(track_id, {}).get("is_stacking", False))
            if should_start_stack or already_stacking:
                if should_start_stack:
                    gate_frames += 1
                stack = movinet_stacks.setdefault(
                    track_id,
                    {
                        "is_stacking": False,
                        "frame_buffer": [],
                        "reference_center": center_xyxy(crowd_box),
                        "max_bbox": crowd_box.copy(),
                    },
                )
                if not stack["is_stacking"]:
                    stack["frame_buffer"] = []
                    stack["is_stacking"] = True
                    stack["reference_center"] = center_xyxy(crowd_box)
                    stack["max_bbox"] = crowd_box.copy()

                stack["reference_center"] = center_xyxy(crowd_box)
                max_bbox = stack["max_bbox"].copy()
                max_bbox[2] = max_bbox[0] + max(float(max_bbox[2] - max_bbox[0]), float(crowd_box[2] - crowd_box[0]))
                max_bbox[3] = max_bbox[1] + max(float(max_bbox[3] - max_bbox[1]), float(crowd_box[3] - crowd_box[1]))
                stack["max_bbox"] = max_bbox
                stack["frame_buffer"].append({"frame": frame_rgb.copy(), "crowd_box": crowd_box.copy(), "has_crowd": True})
                if len(stack["frame_buffer"]) > int(args.clip_length):
                    stack["frame_buffer"] = stack["frame_buffer"][-int(args.clip_length) :]

        current_track_ids = set(group_track_ids)
        for stack_track_id, stack in list(movinet_stacks.items()):
            if stack_track_id in current_track_ids:
                continue
            if not stack.get("is_stacking", False) or len(stack["frame_buffer"]) >= int(args.clip_length):
                continue
            ref_center = stack["reference_center"]
            max_bbox = stack["max_bbox"]
            bw = max(1.0, float(max_bbox[2] - max_bbox[0]))
            bh = max(1.0, float(max_bbox[3] - max_bbox[1]))
            ref_box = clamp_xyxy(
                np.array(
                    [
                        ref_center[0] - bw / 2.0,
                        ref_center[1] - bh / 2.0,
                        ref_center[0] + bw / 2.0,
                        ref_center[1] + bh / 2.0,
                    ],
                    dtype=np.float32,
                ),
                frame_w,
                frame_h,
            )
            stack["frame_buffer"].append({"frame": frame_rgb.copy(), "crowd_box": ref_box, "has_crowd": False})

        for stack_track_id, stack in list(movinet_stacks.items()):
            if len(stack.get("frame_buffer", [])) >= int(args.clip_length):
                t_crop = time.perf_counter()
                completed_sequence = build_movinet_sequence_cpp_like(
                    stack["frame_buffer"][: int(args.clip_length)],
                    frame_w,
                    frame_h,
                    int(args.size),
                    int(args.clip_length),
                )
                cpp_crop_ms += (time.perf_counter() - t_crop) * 1000.0
                completed_stacks += 1
                del movinet_stacks[stack_track_id]
                break

        gate_ms.append((time.perf_counter() - t0) * 1000.0)
        if completed_sequence is not None:
            break

    full_clip_gate_fallback = False
    allow_full_clip_on_gate = bool(args.full_clip_on_gate)
    gate_passed = bool(person_kinetic_pass) if bool(args.kinetic_gate) else gate_frames > 0
    if completed_sequence is None and allow_full_clip_on_gate and gate_passed and len(full_clip_snapshots) >= int(args.clip_length):
        t_crop = time.perf_counter()
        completed_sequence = build_movinet_sequence_cpp_like(
            full_clip_snapshots[: int(args.clip_length)],
            frame_w,
            frame_h,
            int(args.size),
            int(args.clip_length),
        )
        cpp_crop_ms += (time.perf_counter() - t_crop) * 1000.0
        completed_stacks = 1
        full_clip_gate_fallback = True

    return {
        "status": "success" if completed_sequence is not None else "no_complete_kinematic_stack",
        "used_full_clip_gate_fallback": full_clip_gate_fallback,
        "detector_mode": detector_mode,
        "pipeline_mode": "person_kinetic_full_clip" if bool(args.kinetic_gate) else "crowd_full_clip",
        "sequence": completed_sequence,
        "person_frames": person_frames,
        "crowd_frames": crowd_frames,
        "gate_frames": gate_frames,
        "person_kinetic_frames": person_kinetic_frames,
        "completed_stacks": completed_stacks,
        "yolo_track_ms": float(sum(yolo_track_ms)),
        "yolo_inference_only_ms": float(sum(yolo_inference_only_ms)),
        "bytetrack_ms": float(sum(bytetrack_ms)),
        "hdbscan_ms": float(sum(hdbscan_ms)),
        "gate_ms": float(sum(gate_ms)),
        "cpp_crop_ms": float(cpp_crop_ms),
    }


def infer_m3(model: torch.nn.Module, sequence_rgb: np.ndarray, args: argparse.Namespace, device: torch.device) -> tuple[int, float, float]:
    tensor = make_clip_tensor(sequence_rgb, int(args.clip_length)).float().div_(255.0).unsqueeze(0)
    expected_shape = (1, 3, int(args.clip_length), int(args.size), int(args.size))
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(f"Expected M3 tensor shape {expected_shape}, got {tuple(tensor.shape)}")
    tensor = tensor.to(device, non_blocking=True)
    cuda_sync(device)
    start = time.perf_counter()
    with torch.inference_mode():
        with torch.amp.autocast(device_type="cuda", enabled=bool(args.amp) and device.type == "cuda"):
            logits = model(tensor)
        cuda_sync(device)
        if hasattr(model, "clean_activation_buffers"):
            model.clean_activation_buffers()
        probs = F.softmax(logits.float(), dim=1)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    score = float(probs[0, 1].detach().cpu().item())
    if not math.isfinite(score):
        raise RuntimeError(f"Non-finite M3 score: {score}")
    pred = int(score >= float(args.threshold))
    return pred, score, elapsed_ms


def mean_or_zero(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def peak_vram_mb(device: torch.device) -> float | str:
    if device.type != "cuda":
        return "not_available"
    return float(torch.cuda.max_memory_allocated(device) / (1024**2))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerow(summary)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark M3 cpp-stack pipeline from M1 t50 cache.")
    parser.add_argument("--cache-root", type=Path, default=Path("result/gpu_cache"))
    parser.add_argument("--m1-profile", default="wholeframe_rgb_t50_224")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Frozen M1 manifest override; cache_path remains resolved against --cache-root.",
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--movinet-root", type=Path, default=Path("result/movinet_cppstack_t50_experiments"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("result/m3_cppstack_pipeline_benchmark"))
    parser.add_argument("--clip-length", type=int, default=50)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--person-model", default="yolo11n.pt")
    parser.add_argument("--detector-device", default="0")
    parser.add_argument("--detector-imgsz", type=int, default=640)
    parser.add_argument("--yolo-batch-size", type=int, default=50)
    parser.add_argument(
        "--yolo-batch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run YOLO person detection for the 50 cached frames in one batched predict call.",
    )
    parser.add_argument("--half", action="store_true", help="Use FP16 detector inference.")
    parser.add_argument("--amp", action="store_true", help="Use AMP for MoViNet inference.")
    parser.add_argument("--person-conf", type=float, default=0.25)
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--cluster-min-pts", type=int, default=2)
    parser.add_argument(
        "--hdbscan-epsilon",
        type=float,
        default=0.0,
        help="HDBSCAN cluster_selection_epsilon in pixels; larger values merge nearby person clusters more easily.",
    )
    parser.add_argument(
        "--crowd-fallback-min-persons",
        type=int,
        default=0,
        help="If HDBSCAN finds no crowd, treat all detected persons as one crowd when at least N persons are present. 0 disables.",
    )
    parser.add_argument("--iou-gate", type=float, default=0.85)
    parser.add_argument("--velocity-gate", type=float, default=0.05)
    parser.add_argument("--kappa-frames", type=int, default=2)
    parser.add_argument(
        "--person-match-dist",
        type=float,
        default=1.5,
        help="Maximum normalized center distance for matching person boxes between adjacent frames when YOLO IDs are unavailable.",
    )
    parser.add_argument(
        "--kinetic-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require kinetic motion before starting a cpp-stack. Use --no-kinetic-gate to start on any crowd track.",
    )
    parser.add_argument("--crowd-retain-frames", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Rows to skip after split/exclusion filtering; enables resumable artifact batches.",
    )
    parser.add_argument(
        "--exclude-video-ids",
        nargs="*",
        default=[],
        help="Video IDs to exclude from the selected manifest rows.",
    )
    parser.add_argument(
        "--exclude-video-ids-file",
        type=Path,
        default=None,
        help="Optional text file with one video_id per line to exclude.",
    )
    parser.add_argument(
        "--full-clip-on-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If a kinetic gate fires but no post-gate 50-frame stack completes, crop the full cached 50-frame clip and run M3.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if HDBSCAN_BACKEND is None:
        raise RuntimeError("hdbscan is required for this cpp-stack benchmark. Install with: pip install hdbscan")
    device = device_from_arg(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    manifest_path = args.manifest or (args.cache_root / args.m1_profile / "manifest.csv")
    if not manifest_path.exists():
        raise FileNotFoundError(f"M1 manifest not found: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    if args.split != "all":
        manifest = manifest[manifest["split"] == args.split].reset_index(drop=True)
    excluded_video_ids = {str(x).strip() for x in args.exclude_video_ids if str(x).strip()}
    if args.exclude_video_ids_file is not None:
        if not args.exclude_video_ids_file.exists():
            raise FileNotFoundError(f"Exclude video IDs file not found: {args.exclude_video_ids_file}")
        excluded_video_ids.update(
            line.strip()
            for line in args.exclude_video_ids_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
    if excluded_video_ids:
        if "video_id" not in manifest.columns:
            raise ValueError(f"Manifest has no video_id column: {manifest_path}")
        manifest = manifest[~manifest["video_id"].astype(str).isin(excluded_video_ids)].reset_index(drop=True)
    if args.offset:
        manifest = manifest.iloc[int(args.offset):].reset_index(drop=True)
    if args.limit is not None:
        manifest = manifest.head(int(args.limit)).reset_index(drop=True)
    if manifest.empty:
        raise RuntimeError(f"No rows selected from {manifest_path} split={args.split}")

    checkpoint_path = args.checkpoint or (args.movinet_root / "variant_M3" / f"t{args.clip_length}" / "best.pt")
    model = create_model(device)
    load_state(model, checkpoint_path, device)
    model.eval()

    rows: list[dict[str, Any]] = []
    true_labels: list[int] = []
    pred_labels: list[int] = []
    total_start = time.perf_counter()

    profile_dir = args.cache_root / args.m1_profile
    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Benchmark M3 cpp-stack"):
        per_start = time.perf_counter()
        cache_path = resolve_cache_path(args.cache_root, profile_dir, str(row["cache_path"]))
        true_label = int(row["label"])
        pred_label = 0
        score_violence = 0.0
        m3_ms = 0.0
        skip_reason = ""

        frames_rgb = load_cached_rgb_clip(cache_path, int(args.clip_length))
        pipe = run_cppstack_on_cached_frames(frames_rgb, args)

        if pipe["status"] == "success":
            pred_label, score_violence, m3_ms = infer_m3(model, pipe["sequence"], args, device)
            preprocess_status = "success_full_clip_on_gate" if bool(pipe["used_full_clip_gate_fallback"]) else "success"
        else:
            preprocess_status = "skipped_normal"
            skip_reason = str(pipe["status"])

        total_ms = (time.perf_counter() - per_start) * 1000.0
        true_labels.append(true_label)
        pred_labels.append(pred_label)
        rows.append(
            {
                "video_id": str(row.get("video_id", "")),
                "true_label": true_label,
                "pred_label": pred_label,
                "score_violence": score_violence,
                "preprocess_status": preprocess_status,
                "skip_reason": skip_reason,
                "pipeline_mode": str(pipe["pipeline_mode"]),
                "detector_mode": str(pipe["detector_mode"]),
                "person_frames": int(pipe["person_frames"]),
                "person_kinetic_frames": int(pipe["person_kinetic_frames"]),
                "crowd_frames": int(pipe["crowd_frames"]),
                "gate_frames": int(pipe["gate_frames"]),
                "completed_stacks": int(pipe["completed_stacks"]),
                "total_ms": total_ms,
                "yolo_track_ms": float(pipe["yolo_track_ms"]),
                "yolo_inference_only_ms": float(pipe["yolo_inference_only_ms"]),
                "bytetrack_ms": float(pipe["bytetrack_ms"]),
                "hdbscan_ms": float(pipe["hdbscan_ms"]),
                "gate_ms": float(pipe["gate_ms"]),
                "cpp_crop_ms": float(pipe["cpp_crop_ms"]),
                "m3_ms": m3_ms,
                "cache_path": str(row.get("cache_path", "")),
                "source_video": str(row.get("source_video", "")),
            }
        )

    elapsed = time.perf_counter() - total_start
    y_true = np.array(true_labels)
    y_pred = np.array(pred_labels)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(x) for x in cm.ravel()]
    m3_calls = sum(1 for r in rows if int(r["completed_stacks"]) > 0)
    skipped_normal = len(rows) - m3_calls

    summary = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "num_samples": int(len(rows)),
        "m3_calls": int(m3_calls),
        "skipped_normal": int(skipped_normal),
        "pipeline_success_rate": float(m3_calls / len(rows)) if rows else 0.0,
        "fps_videos_per_sec": float(len(rows) / elapsed) if elapsed > 0 else 0.0,
        "latency_mean_ms_per_video": mean_or_zero([float(r["total_ms"]) for r in rows]),
        "yolo_track_ms": mean_or_zero([float(r["yolo_track_ms"]) for r in rows]),
        "yolo_inference_only_ms": mean_or_zero([float(r["yolo_inference_only_ms"]) for r in rows]),
        "bytetrack_ms": mean_or_zero([float(r.get("bytetrack_ms", 0.0)) for r in rows]),
        "hdbscan_ms": mean_or_zero([float(r["hdbscan_ms"]) for r in rows]),
        "kinetic_gate_ms": mean_or_zero([float(r["gate_ms"]) for r in rows]),
        "cpp_crop_ms": mean_or_zero([float(r["cpp_crop_ms"]) for r in rows]),
        "m3_inference_ms": mean_or_zero([float(r["m3_ms"]) for r in rows if float(r["m3_ms"]) > 0]),
        "peak_vram_mb": peak_vram_mb(device),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_video.csv", rows, PER_VIDEO_FIELDS)
    write_summary_csv(args.output_dir / "summary.csv", summary)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                **summary,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "m1_manifest": str(manifest_path),
                "m1_manifest_sha256": sha256_file(manifest_path),
                "split": args.split,
                "clip_length": int(args.clip_length),
                "device": str(device),
                "excluded_video_ids": sorted(excluded_video_ids),
                "excluded_count": len(excluded_video_ids),
                "offset": int(args.offset),
                "yolo_batch": bool(args.yolo_batch),
                "yolo_batch_size": int(args.yolo_batch_size),
                "person_model": str(args.person_model),
                "person_model_sha256": (
                    sha256_file(Path(args.person_model)) if Path(args.person_model).exists() else None
                ),
                "detector_device": str(args.detector_device),
                "detector_imgsz": int(args.detector_imgsz),
                "person_conf": float(args.person_conf),
                "tracker": str(args.tracker),
                "detector_half": bool(args.half),
                "movinet_amp": bool(args.amp),
                "label_conditioned_pipeline": False,
                "kinetic_gate": bool(args.kinetic_gate),
                "iou_gate": float(args.iou_gate),
                "velocity_gate": float(args.velocity_gate),
                "kappa_frames": int(args.kappa_frames),
                "crowd_retain_frames": int(args.crowd_retain_frames),
                "full_clip_on_gate": bool(args.full_clip_on_gate),
                "classification_threshold": float(args.threshold),
                "cluster_min_pts": int(args.cluster_min_pts),
                "crowd_fallback_min_persons": int(args.crowd_fallback_min_persons),
                "hdbscan_epsilon": float(args.hdbscan_epsilon),
                "person_match_dist": float(args.person_match_dist),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

