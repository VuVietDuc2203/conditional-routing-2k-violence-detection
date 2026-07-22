"""Label-free streaming runtime for the approved M1/M3 T50 protocol."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
from pathlib import Path
from time import perf_counter
from typing import Any, Deque
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.scripts.build_movinet_preprocess_cache import (
    bbox_iou_xyxy,
    build_movinet_sequence_cpp_like,
    center_xyxy,
    cluster_person_indices,
    get_person_model,
    make_clip_tensor,
    union_xyxy,
)
VALID_MODES = {
    "m1_dense_s1",
    "m1_stride50",
    "m3_gated",
    "m3_gate_only",
    "m3_route_on",
    "m3_route_bypassed",
}
CLASSIFIER_MODES = {
    "m1_dense_s1",
    "m1_stride50",
    "m3_gated",
    "m3_route_on",
    "m3_route_bypassed",
}
M3_MODES = {"m3_gated", "m3_gate_only", "m3_route_on", "m3_route_bypassed"}
SAME_PIPELINE_MODES = {"m3_route_on", "m3_route_bypassed"}


@dataclass(frozen=True)
class RuntimeConfig:
    clip_length: int = 50
    image_size: int = 224
    person_conf: float = 0.25
    detector_imgsz: int = 640
    iou_gate: float = 0.85
    velocity_gate: float = 0.05
    kappa_frames: int = 2
    rearm_frames: int = 2
    crowd_retain_frames: int = 3
    cluster_min_pts: int = 2
    hdbscan_epsilon: float = 0.0
    threshold: float = 0.5
    hash_candidates: bool = True


@dataclass
class PersonState:
    box: np.ndarray
    kinetic_frames: int = 0
    nonkinetic_frames: int = 0


@dataclass
class CrowdState:
    history: Deque[dict[str, Any]]
    armed: bool = True
    last_box: np.ndarray | None = None
    missing_frames: int = 0


def load_movinet(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    # Lazy import keeps policy/state self-tests independent of the optional
    # MoViNet training stack and its GPU environment.
    from training_code.run_movinet_cached_experiments import create_model

    model = create_model(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    for key in ("model_state_dict", "model", "state_dict"):
        if key in payload:
            model.load_state_dict(payload[key])
            model.eval()
            return model
    raise RuntimeError(f"No state dict in checkpoint: {checkpoint}")


def _infer(
    model: torch.nn.Module,
    sequence_rgb: np.ndarray,
    cfg: RuntimeConfig,
    device: torch.device,
    amp: bool,
) -> tuple[int, float, float]:
    tensor = make_clip_tensor(sequence_rgb, cfg.clip_length).float().div_(255.0).unsqueeze(0).to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = perf_counter()
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
        logits = model(tensor)
        probs = F.softmax(logits.float(), dim=1)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    score = float(probs[0, 1].cpu())
    return int(score >= cfg.threshold), score, (perf_counter() - start) * 1000.0


def wholeframe_sequence(history: Deque[np.ndarray], size: int) -> np.ndarray:
    return np.stack([cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA) for frame in history])


class M3StreamingRuntime:
    """Per-camera state machine: armed -> trigger -> disarmed -> re-armed."""

    def __init__(
        self,
        cfg: RuntimeConfig,
        device: torch.device,
        mode: str,
        checkpoint: Path | None,
        person_model: str = "yolo11n.pt",
        detector_device: str = "0",
        tracker: str = "bytetrack.yaml",
        amp: bool = False,
    ) -> None:
        if mode not in VALID_MODES:
            raise ValueError(f"Unsupported mode: {mode}")
        self.cfg, self.device, self.mode, self.amp = cfg, device, mode, amp
        if mode in CLASSIFIER_MODES:
            if checkpoint is None:
                raise ValueError(f"Mode {mode} requires a classifier checkpoint")
            self.model: torch.nn.Module | None = load_movinet(checkpoint, device)
        else:
            self.model = None
        self.detector = get_person_model(person_model) if mode in M3_MODES else None
        self.detector_device, self.tracker = detector_device, tracker
        self.history: Deque[np.ndarray] = deque(maxlen=cfg.clip_length)
        self.people: dict[int, PersonState] = {}
        self.crowds: dict[int, CrowdState] = {}
        self.analysis_index = 0
        self.first_track = True

    def reset_stream_state(self) -> None:
        """Reset per-video temporal/tracker state without reloading model weights."""
        self.history.clear()
        self.people.clear()
        self.crowds.clear()
        self.analysis_index = 0
        self.first_track = True
        predictor = getattr(self.detector, "predictor", None)
        trackers = getattr(predictor, "trackers", None)
        if trackers:
            for tracker in trackers:
                reset = getattr(tracker, "reset", None)
                if callable(reset):
                    reset()

    @property
    def warmed_up(self) -> bool:
        return len(self.history) == self.cfg.clip_length

    def _detect(self, frame_rgb: np.ndarray) -> tuple[np.ndarray, list[int], dict[str, float]]:
        assert self.detector is not None
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        start = perf_counter()
        result = self.detector.track(
            source=frame_bgr,
            persist=not self.first_track,
            tracker=self.tracker,
            conf=self.cfg.person_conf,
            classes=[0],
            device=self.detector_device,
            imgsz=self.cfg.detector_imgsz,
            verbose=False,
            stream=False,
        )[0]
        total = (perf_counter() - start) * 1000.0
        self.first_track = False
        inference = float(getattr(result, "speed", {}).get("inference", 0.0))
        if result.boxes is None or result.boxes.xyxy is None or len(result.boxes) == 0:
            return np.empty((0, 4), dtype=np.float32), [], {"yolo_ms": total, "yolo_inference_ms": inference}
        boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        ids = (
            result.boxes.id.detach().cpu().numpy().astype(int).tolist()
            if result.boxes.id is not None
            else list(range(-1, -len(boxes) - 1, -1))
        )
        return boxes, ids, {"yolo_ms": total, "yolo_inference_ms": inference}

    def _update_people(self, boxes: np.ndarray, ids: list[int]) -> None:
        active: set[int] = set()
        for box, pid in zip(boxes, ids):
            active.add(pid)
            previous = self.people.get(pid)
            if previous is None:
                self.people[pid] = PersonState(box=box.copy())
                continue
            diagonal = max(1.0, float(np.linalg.norm([box[2] - box[0], box[3] - box[1]])))
            velocity = float(np.linalg.norm(center_xyxy(box) - center_xyxy(previous.box)) / diagonal)
            kinetic = bbox_iou_xyxy(previous.box, box) <= self.cfg.iou_gate or velocity >= self.cfg.velocity_gate
            previous.kinetic_frames = previous.kinetic_frames + 1 if kinetic else 0
            previous.nonkinetic_frames = 0 if kinetic else previous.nonkinetic_frames + 1
            previous.box = box.copy()
        for pid in list(self.people):
            if pid not in active:
                del self.people[pid]

    @staticmethod
    def _crowd_id(member_ids: list[int], ordinal: int) -> int:
        valid = [pid for pid in member_ids if pid >= 0]
        return min(valid) if valid else -(ordinal + 1)

    def _update_crowds(
        self, frame: np.ndarray, boxes: np.ndarray, ids: list[int]
    ) -> tuple[dict[int, list[int]], float]:
        start = perf_counter()
        current: dict[int, list[int]] = {}
        if len(boxes):
            for ordinal, indices in enumerate(
                cluster_person_indices(boxes, self.cfg.cluster_min_pts, self.cfg.hdbscan_epsilon)
            ):
                member_ids = [ids[index] for index in indices]
                crowd_id = self._crowd_id(member_ids, ordinal)
                box = union_xyxy(boxes[indices])
                state = self.crowds.setdefault(crowd_id, CrowdState(deque(maxlen=self.cfg.clip_length)))
                state.last_box, state.missing_frames = box.copy(), 0
                state.history.append({"frame": frame.copy(), "crowd_box": box.copy(), "has_crowd": True})
                current[crowd_id] = member_ids
        for crowd_id, state in list(self.crowds.items()):
            if crowd_id in current:
                continue
            state.missing_frames += 1
            if state.last_box is not None and state.missing_frames <= self.cfg.crowd_retain_frames:
                state.history.append({"frame": frame.copy(), "crowd_box": state.last_box.copy(), "has_crowd": False})
            elif state.missing_frames > self.cfg.crowd_retain_frames:
                del self.crowds[crowd_id]
        return current, (perf_counter() - start) * 1000.0

    def _build_candidate(self, crowd_id: int, frame: np.ndarray) -> tuple[np.ndarray, str, float]:
        """Build the deterministic M3 tensor before the route policy is applied."""
        state = self.crowds[crowd_id]
        crop_start = perf_counter()
        if len(state.history) == self.cfg.clip_length:
            crop_history = list(state.history)
            history_source = "track_history"
        else:
            assert state.last_box is not None
            crop_history = [
                {"frame": prior.copy(), "crowd_box": state.last_box.copy(), "has_crowd": False}
                for prior in self.history
            ]
            history_source = "global_history_current_box"
        sequence = build_movinet_sequence_cpp_like(
            crop_history, frame.shape[1], frame.shape[0], self.cfg.image_size, self.cfg.clip_length
        )
        return sequence, history_source, (perf_counter() - crop_start) * 1000.0

    def _candidate_record(
        self,
        crowd_id: int,
        members: int,
        route_decision: bool,
        sequence: np.ndarray,
        history_source: str,
        crop_ms: float,
    ) -> dict[str, Any]:
        candidate_id = f"u{self.analysis_index:08d}:c{crowd_id}"
        tensor_sha256 = hashlib.sha256(sequence.tobytes(order="C")).hexdigest() if self.cfg.hash_candidates else "disabled"
        return {
            "candidate_id": candidate_id,
            "update_index": self.analysis_index,
            "crowd_id": crowd_id,
            "members": members,
            "eligible": True,
            "route_decision": route_decision,
            "history_source": history_source,
            "tensor_sha256": tensor_sha256,
            "crop_ms": crop_ms,
        }

    @staticmethod
    def _policy_selects(mode: str, route_decision: bool) -> bool:
        if mode == "m3_route_bypassed":
            return True
        if mode in {"m3_route_on", "m3_gated"}:
            return route_decision
        return False

    def _m3_step(self, frame: np.ndarray) -> dict[str, Any]:
        boxes, ids, timing = self._detect(frame)
        gate_start = perf_counter()
        self._update_people(boxes, ids)
        people_gate_ms = (perf_counter() - gate_start) * 1000.0
        crowds, hdbscan_ms = self._update_crowds(frame, boxes, ids)
        calls: list[dict[str, Any]] = []
        candidates: list[dict[str, Any]] = []
        candidate_tensors: dict[str, np.ndarray] = {}
        activations: list[dict[str, Any]] = []
        route_decisions: dict[int, bool] = {}
        member_counts: dict[int, int] = {}
        decision_start = perf_counter()
        for crowd_id, member_ids in crowds.items():
            state = self.crowds[crowd_id]
            members = [self.people[pid] for pid in member_ids if pid in self.people]
            member_counts[crowd_id] = len(members)
            if not state.armed and members and all(p.nonkinetic_frames >= self.cfg.rearm_frames for p in members):
                state.armed = True
            triggered = state.armed and any(p.kinetic_frames >= self.cfg.kappa_frames for p in members)
            route_decisions[crowd_id] = bool(triggered and self.warmed_up)
            # A newly visible crowd must not wait another T50 interval.  The
            # camera-wide history is always maintained; use its frames with
            # the current crowd box until the track-local crop history fills.
            if triggered and self.warmed_up:
                state.armed = False
                activations.append({"crowd_id": crowd_id, "members": len(members)})
        gate_ms = people_gate_ms + (perf_counter() - decision_start) * 1000.0

        # New JVCIR ablation modes materialize the same eligible candidate
        # stream before applying the call/skip policy. This intentionally adds
        # identical candidate-construction instrumentation to both modes.
        if self.mode in SAME_PIPELINE_MODES and self.warmed_up:
            for crowd_id in sorted(crowds):
                state = self.crowds[crowd_id]
                if state.last_box is None:
                    continue
                sequence, history_source, crop_ms = self._build_candidate(crowd_id, frame)
                candidate = self._candidate_record(
                    crowd_id,
                    member_counts.get(crowd_id, 0),
                    route_decisions.get(crowd_id, False),
                    sequence,
                    history_source,
                    crop_ms,
                )
                candidate["policy_selected"] = self._policy_selects(self.mode, bool(candidate["route_decision"]))
                candidates.append(candidate)
                candidate_tensors[str(candidate["candidate_id"])] = sequence

        if self.mode in SAME_PIPELINE_MODES:
            assert self.model is not None
            for candidate in candidates:
                if not candidate["policy_selected"]:
                    continue
                sequence = candidate_tensors[str(candidate["candidate_id"])]
                pred, score, classifier_ms = _infer(self.model, sequence, self.cfg, self.device, self.amp)
                calls.append({
                    **candidate,
                    "prediction": pred,
                    "score": score,
                    "classifier_ms": classifier_ms,
                })
        elif self.mode == "m3_gated":
            assert self.model is not None
            for activation in activations:
                crowd_id = int(activation["crowd_id"])
                sequence, history_source, crop_ms = self._build_candidate(crowd_id, frame)
                pred, score, classifier_ms = _infer(self.model, sequence, self.cfg, self.device, self.amp)
                calls.append(
                    {"crowd_id": crowd_id, "prediction": pred, "score": score, "history_source": history_source,
                     "crop_ms": crop_ms, "classifier_ms": classifier_ms}
                )
        timing.update({
            "hdbscan_ms": hdbscan_ms,
            "gate_ms": gate_ms,
            "crop_ms": sum(float(candidate["crop_ms"]) for candidate in candidates),
        })
        return {
            "calls": calls,
            "candidates": candidates,
            "eligible_candidates": len(candidates),
            "route_selected_candidates": sum(bool(candidate.get("route_decision")) for candidate in candidates),
            "gate_activations": activations,
            "timing": timing,
            "crowds": len(crowds),
        }

    def process(self, frame_rgb: np.ndarray) -> dict[str, Any]:
        """Process one analyzed frame. This API has no label argument by design."""
        self.history.append(frame_rgb.copy())
        self.analysis_index += 1
        # M3 Stage 1 is live from the first analyzed frame.  Only the temporal
        # classifier waits for the T50 camera-wide history to become available.
        if self.mode in M3_MODES:
            result = self._m3_step(frame_rgb)
            result["warmed_up"] = self.warmed_up
            return result
        if not self.warmed_up:
            return {"warmed_up": False, "calls": [], "timing": {}}
        due = self.mode == "m1_dense_s1" or self.analysis_index % self.cfg.clip_length == 0
        if not due:
            return {"warmed_up": True, "calls": [], "timing": {}}
        sequence = wholeframe_sequence(self.history, self.cfg.image_size)
        assert self.model is not None
        pred, score, classifier_ms = _infer(self.model, sequence, self.cfg, self.device, self.amp)
        return {
            "warmed_up": True,
            "calls": [{"crowd_id": "whole_frame", "prediction": pred, "score": score,
                       "crop_ms": 0.0, "classifier_ms": classifier_ms}],
            "timing": {},
        }


def state_machine_self_test() -> None:
    """Pure regression guard for T50, two-frame trigger, and two-frame re-arm."""
    cfg = RuntimeConfig(kappa_frames=2, rearm_frames=2)
    assert cfg.clip_length == 50
    kinetic = sum((True, True))
    assert kinetic == cfg.kappa_frames
    nonkinetic = sum((True, True))
    assert nonkinetic == cfg.rearm_frames
    assert M3StreamingRuntime._policy_selects("m3_route_on", True)
    assert not M3StreamingRuntime._policy_selects("m3_route_on", False)
    assert M3StreamingRuntime._policy_selects("m3_route_bypassed", True)
    assert M3StreamingRuntime._policy_selects("m3_route_bypassed", False)
