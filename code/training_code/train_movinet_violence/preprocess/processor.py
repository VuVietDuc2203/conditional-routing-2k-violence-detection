from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from collections import Counter, deque

import cv2
import numpy as np

try:
    import hdbscan  # type: ignore
except Exception:  # pragma: no cover
    hdbscan = None

try:
    from ultralytics import YOLO  # type: ignore
except Exception:  # pragma: no cover
    YOLO = None

from .config import PreprocessConfig
from .utils import (
    BBox,
    area_xywh,
    clamp_bbox_xywh,
    crop_and_resize,
    expand_bbox,
    overlap_ratio,
    union_bboxes,
    xyxy_to_xywh,
)

# Type aliases for compatibility with crow_debug.py logic
BBoxXYXY = tuple[float, float, float, float]  # (x1, y1, x2, y2)


@dataclass
class CrowdTrackInfo:
    bbox: BBox
    track_id: int
    frame_idx: int


@dataclass
class StackInfo:
    stack_id: int
    frames: deque[np.ndarray] = field(default_factory=deque)  # cropped+resized frames (172x172)
    original_frames: deque[np.ndarray] = field(default_factory=deque)  # full frames for re-crop
    last_frame_crop_bbox: BBox | None = None
    frame_count: int = 0
    last_update_frame: int = 0
    frame_crowd_bboxes: list[tuple[int, BBox, int]] = field(default_factory=list)  # (local_idx, bbox, track_id)
    frame_crowd_centers: dict[int, tuple[float, float]] = field(default_factory=dict)  # local_idx -> (cx, cy)
    # ByteTrack: person IDs in each frame for each crowd box
    frame_crowd_person_ids: dict[int, list[int]] = field(default_factory=dict)  # local_idx -> [person_id1, person_id2, ...]
    # Person bboxes for each frame (for special-case full-frame resize rule)
    # Stored as list of tuples (bbox, person_id) where person_id can be -1 if not tracked
    frame_person_boxes: dict[int, list[tuple[BBox, int]]] = field(default_factory=dict)  # local_idx -> [(bbox_xywh, person_id), ...]


@dataclass
class ExtractedStack:
    stack_id: int
    label: str  # "violence" or "normal"
    frames: list[np.ndarray]  # 16 frames (BGR) resized to crop_resize_size


class VideoStackProcessor:
    def __init__(self, cfg: PreprocessConfig):
        self.cfg = cfg

        if YOLO is None:
            raise RuntimeError("ultralytics is required. Install with: pip install ultralytics")

        self.person_model = YOLO(cfg.models.person_detector_onnx)
        self.fight_model = YOLO(cfg.models.fight_detector_onnx)

        # state
        self._movinet_stacks: dict[int, StackInfo] = {}
        self._prev_crowd_tracks: list[CrowdTrackInfo] = []
        self._crowd_tracks_history: list[CrowdTrackInfo] = []
        self._crowd_crop_boxes: dict[int, BBox] = {}

        self._next_stack_id: int = 0
        self._next_crowd_track_id: int = 0

        self._crowd_frame_count: int = 0
        self._crowd_alert_active: bool = False
        self._frame_counter: int = 0

        # Cached person detections/tracks from the most recent raw frame
        self._cached_person_boxes: list[BBox] = []
        self._cached_person_centers: list[tuple[float, float]] = []
        self._cached_person_ids: list[int] = []
        
        # Crowd tracking state (based on person IDs, not bbox overlap)
        # {crowd_id: {"person_ids": set, "bbox": BBox, "age": int, "last_seen": int}}
        self._tracked_crowds: dict[int, dict[str, Any]] = {}
        self._crowd_max_age: int = 3  # Giữ crowd tối đa 3 frame nếu không phát hiện

    def reset_state(self) -> None:
        """Reset all tracking state. Call when starting a new video."""
        self._movinet_stacks.clear()
        self._prev_crowd_tracks.clear()
        self._crowd_tracks_history.clear()
        self._crowd_crop_boxes.clear()

        self._next_stack_id = 0
        self._next_crowd_track_id = 0
        self._crowd_frame_count = 0
        self._crowd_alert_active = False
        self._frame_counter = 0
        
        # Reset ByteTrack tracker if using tracking
        if self.cfg.tracking.use_bytetrack:
            # Ultralytics tracker persists automatically, but we can reset by creating new model
            # Actually, tracker state is maintained by ultralytics, so no need to reset here
            pass

        self._cached_person_boxes = []
        self._cached_person_centers = []
        self._cached_person_ids = []
        
        # Reset crowd tracking state
        self._tracked_crowds.clear()

    def update_person_tracks(self, frame_bgr: np.ndarray) -> tuple[list[BBox], list[tuple[float, float]], list[int]]:
        """Update ByteTrack (or plain detection) on a raw frame and cache results.

        Call this for *every* frame of the video to stabilize person IDs,
        then call `process_sampled_frame()` only at 4fps for crowd/stacking logic.
        """
        # Person detection with ByteTrack (if enabled)
        if self.cfg.tracking.use_bytetrack:
            track_kwargs = {
                "source": frame_bgr,
                "verbose": False,
                "conf": float(self.cfg.person_detection.conf_threshold),
                "device": self.cfg.runtime.device,
                "show": bool(self.cfg.runtime.show),
                "persist": True,
            }
            tracker_str = str(self.cfg.tracking.tracker)
            if tracker_str.endswith((".yaml", ".yml")) or "/" in tracker_str or "\\" in tracker_str:
                track_kwargs["tracker"] = tracker_str
            det = self.person_model.track(**track_kwargs)
        else:
            det = self.person_model.predict(
                source=frame_bgr,
                verbose=False,
                conf=float(self.cfg.person_detection.conf_threshold),
                device=self.cfg.runtime.device,
                show=bool(self.cfg.runtime.show),
            )

        boxes: list[BBox] = []
        centers: list[tuple[float, float]] = []
        person_ids: list[int] = []
        for r in det:
            if r.boxes is None:
                continue
            xyxy = r.boxes.xyxy.detach().cpu().numpy()
            cls = r.boxes.cls.detach().cpu().numpy().astype(int)
            track_ids = None
            if r.boxes.id is not None:
                track_ids = r.boxes.id.detach().cpu().numpy().astype(int)

            for idx, ((x1, y1, x2, y2), c) in enumerate(zip(xyxy, cls)):
                if int(c) != 0:
                    continue
                b = xyxy_to_xywh(float(x1), float(y1), float(x2), float(y2))
                boxes.append(b)
                centers.append((b[0] + b[2] / 2.0, b[1] + b[3] / 2.0))
                if track_ids is not None and idx < len(track_ids):
                    person_ids.append(int(track_ids[idx]))
                else:
                    person_ids.append(-1)

        self._cached_person_boxes = boxes
        self._cached_person_centers = centers
        self._cached_person_ids = person_ids
        return boxes, centers, person_ids

    def process_sampled_frame(self, frame_bgr: np.ndarray, camera_id: str) -> dict[str, Any]:
        """Process a sampled frame (4fps) using cached person tracks (preferred)."""
        if not self._cached_person_boxes and not self._cached_person_centers:
            boxes, centers, person_ids = self.update_person_tracks(frame_bgr)
        else:
            boxes, centers, person_ids = (
                self._cached_person_boxes,
                self._cached_person_centers,
                self._cached_person_ids,
            )
        return self._process_with_persons(frame_bgr, camera_id, boxes, centers, person_ids)

    def _process_fight_results(self, yolo_results: Any) -> tuple[bool, float]:
        """Return (has_fight, confidence)."""
        fight_thresh = self.cfg.fight_detection.conf_threshold
        best_normal = 0.0
        try:
            for r in yolo_results:
                if r.boxes is None:
                    continue
                cls = r.boxes.cls.detach().cpu().numpy().astype(int)
                conf = r.boxes.conf.detach().cpu().numpy()
                for c, p in zip(cls, conf):
                    if c == 1 and float(p) >= fight_thresh:
                        return True, float(p)
                    if c == 0:
                        best_normal = max(best_normal, float(p))
        except Exception:
            return False, 0.0
        return False, best_normal

    def _iou_bbox_xyxy(self, box1: BBoxXYXY, box2: BBoxXYXY) -> float:
        """Tính IoU (Intersection over Union) của 2 bbox để kiểm tra overlap."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Tính intersection
        inter_x1 = max(x1_1, x1_2)
        inter_y1 = max(y1_1, y1_2)
        inter_x2 = min(x2_1, x2_2)
        inter_y2 = min(y2_1, y2_2)
        
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0

    def _is_pair_crowd(
        self,
        center1: tuple[float, float],
        center2: tuple[float, float],
        box1: BBoxXYXY,
        box2: BBoxXYXY,
    ) -> bool:
        """
        Kiểm tra 2 người có phải là 1 vùng tụ tập không.
        
        Logic:
        - Nếu có overlap (IoU > 0) → là 1 vùng tụ tập
        - Nếu khoảng cách giữa 2 tâm < 2 lần chiều rộng bbox lớn nhất → là 1 vùng tụ tập
        - Ngược lại → không phải 1 vùng tụ tập
        """
        # Kiểm tra overlap
        if self._iou_bbox_xyxy(box1, box2) > 0.0:
            return True
        
        # Tính khoảng cách giữa 2 tâm
        cx1, cy1 = center1
        cx2, cy2 = center2
        distance = np.sqrt((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2)
        
        # Tính chiều rộng lớn nhất của 2 bbox
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        width1 = x2_1 - x1_1
        width2 = x2_2 - x1_2
        max_width = max(width1, width2)
        
        # Nếu khoảng cách < 2 lần chiều rộng lớn nhất → là 1 vùng tụ tập
        return distance < max_width * 2.0

    def _cluster_small_group(
        self,
        centers: list[tuple[float, float]],
        person_boxes_xyxy: list[BBoxXYXY],
        debug: bool = False,
    ) -> list[list[int]]:
        """
        Xử lý đặc biệt cho nhóm 2-4 người: kiểm tra từng đôi một.
        
        Logic:
        - Kiểm tra tất cả các cặp (2 người)
        - Nếu tất cả các cặp đều là crowd → gộp chung 1 cụm
        - Nếu chỉ 1 số cặp là crowd → tạo cụm riêng cho từng cặp, gộp nếu overlap
        - Nếu không có cặp nào là crowd → trả về [] (KHÔNG gộp tất cả)
        """
        n = len(centers)
        if n < 2 or n > 4:
            return []
        
        # Kiểm tra tất cả các cặp
        pair_crowds: dict[tuple[int, int], bool] = {}  # {(i, j): True/False}
        for i in range(n):
            for j in range(i + 1, n):
                is_crowd = self._is_pair_crowd(centers[i], centers[j], person_boxes_xyxy[i], person_boxes_xyxy[j])
                pair_crowds[(i, j)] = is_crowd
        
        # Kiểm tra xem tất cả các cặp có phải là crowd không
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        all_pairs_are_crowd = all(pair_crowds[pair] for pair in all_pairs)
        
        if all_pairs_are_crowd:
            # Tất cả các cặp đều là crowd → gộp chung 1 cụm
            return [list(range(n))]
        
        # Chỉ một số cặp là crowd → tạo cụm riêng cho từng cặp
        clusters = []
        for (i, j), is_crowd in pair_crowds.items():
            if is_crowd:
                clusters.append([i, j])
        
        if not clusters:
            # Không có cặp nào là crowd → không phát hiện crowd
            return []
        
        # Gộp các cụm có overlap (cùng ít nhất 1 người)
        merged_clusters = []
        used = set()
        
        for cluster in clusters:
            cluster_set = set(cluster)
            # Tìm cụm đã merge có overlap với cluster này
            found = False
            for i, merged_set in enumerate(merged_clusters):
                if cluster_set & merged_set:  # Có overlap
                    merged_clusters[i] = merged_set | cluster_set
                    found = True
                    break
            if not found:
                merged_clusters.append(cluster_set)
        
        # Convert set về list và sort
        result = [sorted(list(cluster)) for cluster in merged_clusters]
        return result

    def _cluster_crowds(
        self, 
        centers: list[tuple[float, float]], 
        person_boxes: list[BBox] | None = None
    ) -> list[list[int]]:
        """Trả về các cụm crowd, mỗi cụm là list index person."""
        if not centers:
            return []

        min_pts = max(2, int(self.cfg.clustering.cluster_min_pts))
        if len(centers) < min_pts:
            return []

        # Nếu <= 4 người: dùng hardcode logic (cluster_small_group)
        if len(centers) <= 4:
            if person_boxes is not None and len(person_boxes) == len(centers):
                # Convert BBox (x,y,w,h) to BBoxXYXY (x1,y1,x2,y2)
                person_boxes_xyxy: list[BBoxXYXY] = []
                for b in person_boxes:
                    x, y, w, h = b
                    person_boxes_xyxy.append((float(x), float(y), float(x + w), float(y + h)))
                return self._cluster_small_group(centers, person_boxes_xyxy, debug=False)
            return []  # Không có person_boxes → không thể dùng hardcode
        
        # Nếu > 3 người: LUÔN dùng HDBSCAN
        if hdbscan is None:
            # Nếu không có hdbscan → fallback: coi tất cả là 1 cụm
            return [list(range(len(centers)))]
        
        pts = np.asarray(centers, dtype=np.float32)
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_pts,
            min_samples=max(1, min_pts - 1),
            metric="euclidean",
            cluster_selection_method="eom",
        )
        labels = clusterer.fit_predict(pts)

        clusters: dict[int, list[int]] = {}
        for idx, lab in enumerate(labels):
            if int(lab) < 0:
                continue
            clusters.setdefault(int(lab), []).append(idx)

        result = [idxs for idxs in clusters.values() if len(idxs) >= min_pts]
        
        # Trả về kết quả HDBSCAN (có thể là [] nếu không tìm được cụm nào)
        return result

    def _find_or_create_stack(self, bbox: BBox) -> int:
        overlap_th = float(self.cfg.tracking.overlap_threshold)
        for stack_id, stack in self._movinet_stacks.items():
            if stack.frame_count <= 0 or stack.last_frame_crop_bbox is None:
                continue
            if overlap_ratio(stack.last_frame_crop_bbox, bbox) > overlap_th:
                return stack_id

        self._next_stack_id += 1
        stack_id = self._next_stack_id
        self._movinet_stacks[stack_id] = StackInfo(
            stack_id=stack_id,
            last_frame_crop_bbox=bbox,
            last_update_frame=self._frame_counter,
        )
        return stack_id

    def _assign_persons_to_crowd(self, person_boxes: list[BBox], person_ids: list[int], crowd_bbox: BBox) -> list[int]:
        """Assign person IDs to a crowd box if person bbox overlaps with crowd bbox."""
        assigned_ids: list[int] = []
        for pbox, pid in zip(person_boxes, person_ids):
            if pid < 0:  # Skip invalid IDs
                continue
            # Check if person bbox overlaps with crowd bbox
            if overlap_ratio(pbox, crowd_bbox) > 0.1:  # 10% overlap threshold
                assigned_ids.append(pid)
        return assigned_ids

    def _compute_crop_centers(self, stack: StackInfo, max_crop_bbox: BBox) -> list[tuple[float, float]]:
        centers: list[tuple[float, float]] = []
        fallback = (max_crop_bbox[0] + max_crop_bbox[2] / 2.0, max_crop_bbox[1] + max_crop_bbox[3] / 2.0)

        available = sorted(stack.frame_crowd_centers.items(), key=lambda x: x[0])  # (idx, (cx,cy))
        available_idxs = [i for i, _ in available]

        for i in range(self.cfg.stacking.sequence_frames):
            if i in stack.frame_crowd_centers:
                centers.append(stack.frame_crowd_centers[i])
                continue

            before = [j for j in available_idxs if j < i]
            after = [j for j in available_idxs if j > i]
            if before and after:
                ib = before[-1]
                ia = after[0]
                cb = stack.frame_crowd_centers[ib]
                ca = stack.frame_crowd_centers[ia]
                denom = float(ia - ib) if ia != ib else 1.0
                t = float(i - ib) / denom
                centers.append((cb[0] + t * (ca[0] - cb[0]), cb[1] + t * (ca[1] - cb[1])))
            elif before:
                centers.append(stack.frame_crowd_centers[before[-1]])
            elif after:
                centers.append(stack.frame_crowd_centers[after[0]])
            else:
                centers.append(fallback)

        return centers

    def _compute_group_person_ids(self, stack: StackInfo) -> set[int]:
        """Compute a group of person IDs that have ever been in the same crowd together.

        Build a graph where:
        - Nodes = person IDs
        - Edges = two IDs that appear together in at least one crowd in one frame
        Then take the largest connected component (with optional min_appearances filter).
        """
        if not self.cfg.tracking.use_bytetrack or not stack.frame_crowd_person_ids:
            return set()
        
        # Collect all person IDs across all frames in crowds
        all_person_ids: list[int] = []
        frame_crowd_ids: dict[int, list[int]] = {}
        for local_idx in range(stack.frame_count):
            ids = stack.frame_crowd_person_ids.get(local_idx, [])
            # Filter out invalid IDs (<0)
            ids = [pid for pid in ids if pid >= 0]
            if not ids:
                continue
            frame_crowd_ids[local_idx] = ids
            all_person_ids.extend(ids)
        
        if not all_person_ids:
            return set()
        
        # Count appearances of each person ID
        person_counts = Counter(all_person_ids)
        min_appearances = int(self.cfg.tracking.min_person_appearances)
        
        # Filter: keep only person IDs appearing >= min_appearances times
        valid_ids = {pid for pid, count in person_counts.items() if count >= min_appearances}
        if not valid_ids:
            return set()

        # Build adjacency graph of co-occurrence in crowds
        adj: dict[int, set[int]] = {pid: set() for pid in valid_ids}
        for ids in frame_crowd_ids.values():
            # Only consider valid IDs
            frame_ids = [pid for pid in ids if pid in valid_ids]
            n = len(frame_ids)
            for i in range(n):
                for j in range(i + 1, n):
                    a, b = frame_ids[i], frame_ids[j]
                    adj[a].add(b)
                    adj[b].add(a)

        # Find connected components
        visited: set[int] = set()
        components: list[set[int]] = []

        for pid in valid_ids:
            if pid in visited:
                continue
            comp: set[int] = set()
            stack_ids = [pid]
            visited.add(pid)
            while stack_ids:
                cur = stack_ids.pop()
                comp.add(cur)
                for nb in adj.get(cur, []):
                    if nb not in visited:
                        visited.add(nb)
                        stack_ids.append(nb)
            components.append(comp)

        if not components:
            return set()

        # Choose component with largest size; tie-breaker: larger total appearances
        def comp_score(c: set[int]) -> tuple[int, int]:
            return (len(c), sum(person_counts[pid] for pid in c))

        best_comp = max(components, key=comp_score)
        return best_comp
    
    def _finalize_stack(self, stack: StackInfo) -> ExtractedStack:
        # Analyze person IDs to find a group of people that have ever gathered together
        group_person_ids = self._compute_group_person_ids(stack)

        # Special rule: if a person bbox width is very large (>40% frame width) in >= N frames, skip crowd crop and use full-frame resize
        h0, w0 = stack.original_frames[0].shape[:2]
        ratio = float(self.cfg.stacking.large_person_ratio)
        min_frames = int(self.cfg.stacking.large_person_min_frames)
        large_count = 0
        for local_idx in range(int(self.cfg.stacking.sequence_frames)):
            for pb in stack.frame_person_boxes.get(local_idx, []):
                bbox, _ = pb  # Unpack (bbox, person_id)
                _, _, pw, ph = bbox
                # Chỉ kiểm tra chiều dài (width) của bbox người > ratio * chiều dài frame
                if pw >= ratio * w0:
                    large_count += 1
                    break
            if large_count >= min_frames:
                break

        if large_count >= min_frames:
            # Direct full-frame resize to 172x172 for all 16 frames
            resized = [
                cv2.resize(f, (self.cfg.stacking.crop_resize_size, self.cfg.stacking.crop_resize_size), interpolation=cv2.INTER_LINEAR)
                for f in list(stack.original_frames)
            ]
            # Fight detection & label on resized frames (same as normal flow)
            fight_frame_count = 0
            if self.cfg.runtime.batch_fight_detection and len(resized) > 1:
                try:
                    results = self.fight_model.predict(
                        source=resized,
                        verbose=False,
                        conf=float(self.cfg.fight_detection.conf_threshold),
                        device=self.cfg.runtime.device,
                        show=bool(self.cfg.runtime.show),
                    )
                    for r in results:
                        has_fight, _ = self._process_fight_results([r])
                        if has_fight:
                            fight_frame_count += 1
                except Exception:
                    for f in resized:
                        results = self.fight_model.predict(
                            source=f,
                            verbose=False,
                            conf=float(self.cfg.fight_detection.conf_threshold),
                            device=self.cfg.runtime.device,
                            show=bool(self.cfg.runtime.show),
                        )
                        has_fight, _ = self._process_fight_results(results)
                        if has_fight:
                            fight_frame_count += 1
            else:
                for f in resized:
                    results = self.fight_model.predict(
                        source=f,
                        verbose=False,
                        conf=float(self.cfg.fight_detection.conf_threshold),
                        device=self.cfg.runtime.device,
                        show=bool(self.cfg.runtime.show),
                    )
                    has_fight, _ = self._process_fight_results(results)
                    if has_fight:
                        fight_frame_count += 1

            label = "violence" if fight_frame_count >= int(self.cfg.fight_detection.valid_violence_frames) else "normal"
            return ExtractedStack(stack_id=stack.stack_id, label=label, frames=resized)
        
        # Step 8.1: Build a global crowd bbox from all persons in the group across 16 frames
        max_crowd_bbox: BBox | None = None

        if group_person_ids:
            # Collect all person bboxes whose IDs are in the group
            group_boxes: list[BBox] = []
            for local_idx in range(stack.frame_count):
                for pb in stack.frame_person_boxes.get(local_idx, []):
                    b, pid = pb
                    if pid in group_person_ids:
                        group_boxes.append(b)
            if group_boxes:
                # Union of all boxes to form one global crowd region
                max_crowd_bbox = union_bboxes(group_boxes)

        if max_crowd_bbox is None and stack.frame_crowd_bboxes:
            # Fallback: use union of all crowd bboxes
            all_crowd_boxes = [bbox for _, bbox, _ in stack.frame_crowd_bboxes]
            if all_crowd_boxes:
                max_crowd_bbox = union_bboxes(all_crowd_boxes)

        # Expand the global bbox by crop_scale (1.5x)
        if max_crowd_bbox is not None:
            max_crop_bbox = expand_bbox(max_crowd_bbox, self.cfg.stacking.crop_scale, w0, h0)
        else:
            # Fallback to full frame
            max_crop_bbox = (0, 0, w0, h0)

        # Step 8.3: square crop size (max_crop_bbox is already expanded)
        max_side = max(max_crop_bbox[2], max_crop_bbox[3])
        side = int(max(1, round(max_side)))  # Already expanded, no need to multiply again
        if side > w0:
            side = w0
        if side > h0:
            side = h0

        centers = self._compute_crop_centers(stack, max_crop_bbox)

        cropped: list[np.ndarray] = []
        for frame, (cx, cy) in zip(stack.original_frames, centers):
            x = int(round(cx - side / 2.0))
            y = int(round(cy - side / 2.0))
            x = max(0, min(x, w0 - side))
            y = max(0, min(y, h0 - side))
            crop_bbox = clamp_bbox_xywh((x, y, side, side), w0, h0)
            cropped.append(crop_and_resize(frame, crop_bbox, self.cfg.stacking.crop_resize_size))

        # Step 8.5: fight detection on 16 cropped frames
        fight_frame_count = 0
        
        if self.cfg.runtime.batch_fight_detection and len(cropped) > 1:
            # Batch processing: process all frames together (much faster on GPU)
            try:
                results = self.fight_model.predict(
                    source=cropped,  # list of frames -> batch inference
                    verbose=False,
                    conf=float(self.cfg.fight_detection.conf_threshold),
                    device=self.cfg.runtime.device,
                    show=bool(self.cfg.runtime.show),
                )
                # Process batch results
                for r in results:
                    has_fight, _ = self._process_fight_results([r])
                    if has_fight:
                        fight_frame_count += 1
            except Exception:
                # Fallback to per-frame if batch fails
                for f in cropped:
                    results = self.fight_model.predict(
                        source=f,
                        verbose=False,
                        conf=float(self.cfg.fight_detection.conf_threshold),
                        device=self.cfg.runtime.device,
                        show=bool(self.cfg.runtime.show),
                    )
                    has_fight, _ = self._process_fight_results(results)
                    if has_fight:
                        fight_frame_count += 1
        else:
            # Per-frame processing (original method)
            for f in cropped:
                results = self.fight_model.predict(
                    source=f,
                    verbose=False,
                    conf=float(self.cfg.fight_detection.conf_threshold),
                    device=self.cfg.runtime.device,
                    show=bool(self.cfg.runtime.show),
                )
                has_fight, _ = self._process_fight_results(results)
                if has_fight:
                    fight_frame_count += 1

        label = "violence" if fight_frame_count >= int(self.cfg.fight_detection.valid_violence_frames) else "normal"
        return ExtractedStack(stack_id=stack.stack_id, label=label, frames=cropped)

    def process_frame(self, frame_bgr: np.ndarray, camera_id: str) -> dict[str, Any]:
        """Back-compat: process one sampled frame by updating person tracks on this frame only."""
        boxes, centers, person_ids = self.update_person_tracks(frame_bgr)
        return self._process_with_persons(frame_bgr, camera_id, boxes, centers, person_ids)

    def _process_with_persons(
        self,
        frame_bgr: np.ndarray,
        camera_id: str,
        boxes: list[BBox],
        centers: list[tuple[float, float]],
        person_ids: list[int],
    ) -> dict[str, Any]:
        self._frame_counter += 1

        persons_debug: list[dict[str, Any]] = []
        for b, pid in zip(boxes, person_ids):
            persons_debug.append({"class": "person", "id": int(pid), "bbox_xywh": b})

        # Clustering -> crowd boxes (with person_boxes for hardcode logic)
        clusters = self._cluster_crowds(centers, boxes)
        crowd_boxes: list[BBox] = []
        crowd_person_ids_list: list[list[int]] = []  # person IDs for each crowd
        for idxs in clusters:
            cluster_boxes = [boxes[i] for i in idxs]
            cluster_ids = [person_ids[i] for i in idxs]
            crowd_boxes.append(union_bboxes(cluster_boxes))
            # Collect valid person IDs for this crowd
            valid_ids = [pid for pid in cluster_ids if pid >= 0]
            crowd_person_ids_list.append(valid_ids)

        # Tracking based on person IDs (not bbox overlap)
        current_crowds: list[dict[str, Any]] = []
        current_person_ids_set = set(pid for pid in person_ids if pid >= 0)
        
        for idxs, bbox, crowd_person_ids in zip(clusters, crowd_boxes, crowd_person_ids_list):
            valid_ids = sorted(crowd_person_ids)
            valid_ids_set = set(valid_ids)
            
            if not valid_ids:
                continue  # Skip crowds without valid person IDs
            
            # Tìm crowd từ frame trước có person IDs overlap nhiều nhất
            best_match_id: int | None = None
            best_overlap = 0
            for old_crowd_id, old_crowd in self._tracked_crowds.items():
                old_ids_set = old_crowd["person_ids"]
                overlap = len(valid_ids_set & old_ids_set)
                if overlap > best_overlap and overlap >= 1:  # Ít nhất 1 person ID trùng
                    best_overlap = overlap
                    best_match_id = old_crowd_id
            
            if best_match_id is not None:
                # Match được → dùng crowd_id cũ, update state
                crowd_id = best_match_id
                self._tracked_crowds[crowd_id] = {
                    "person_ids": valid_ids_set,
                    "bbox": bbox,
                    "age": 0,  # Reset age khi phát hiện lại
                    "last_seen": self._frame_counter,
                }
            else:
                # Không match → tạo crowd_id mới
                crowd_id = self._next_crowd_track_id
                self._next_crowd_track_id += 1
                self._tracked_crowds[crowd_id] = {
                    "person_ids": valid_ids_set,
                    "bbox": bbox,
                    "age": 0,
                    "last_seen": self._frame_counter,
                }
            
            current_crowds.append({
                "bbox": bbox,
                "person_ids": valid_ids,
                "crowd_id": crowd_id,
            })
        
        # Keep crowd từ frame trước nếu không phát hiện nhưng vẫn có person IDs tương tự
        for crowd_id, crowd_data in list(self._tracked_crowds.items()):
            if crowd_id in [c["crowd_id"] for c in current_crowds]:
                continue  # Đã được match ở trên
            
            # Kiểm tra xem có person IDs nào của crowd này còn xuất hiện không
            old_ids_set = crowd_data["person_ids"]
            overlap = len(current_person_ids_set & old_ids_set)
            
            if overlap >= 1:  # Có ít nhất 1 person ID còn xuất hiện
                crowd_data["age"] += 1
                if crowd_data["age"] <= self._crowd_max_age:
                    # Keep crowd này, dùng bbox cũ
                    current_crowds.append({
                        "bbox": crowd_data["bbox"],
                        "person_ids": sorted(old_ids_set),
                        "crowd_id": crowd_id,
                    })
                else:
                    # Quá cũ → xóa
                    del self._tracked_crowds[crowd_id]
            else:
                # Không còn person ID nào → xóa
                del self._tracked_crowds[crowd_id]
        
        # Extract group_bboxes and group_track_ids from current_crowds
        group_bboxes = [c["bbox"] for c in current_crowds]
        group_track_ids = [c["crowd_id"] for c in current_crowds]

        # Debug: assign person IDs to each crowd group for this frame
        crowds_debug: list[dict[str, Any]] = []
        for c in current_crowds:
            crowds_debug.append({
                "class": "crowd",
                "id": int(c["crowd_id"]),
                "bbox_xywh": c["bbox"],
                "person_ids": c["person_ids"]
            })

        # Update tracking state
        if group_bboxes:
            self._crowd_frame_count += 1
            if self._crowd_frame_count >= 2:
                self._crowd_alert_active = True

        active_track_ids = set(group_track_ids)
        for tid in list(self._crowd_crop_boxes.keys()):
            if tid not in active_track_ids:
                del self._crowd_crop_boxes[tid]

        # Update history for backward compatibility (but not used for matching)
        self._prev_crowd_tracks.clear()
        for bbox, tid in zip(group_bboxes, group_track_ids):
            info = CrowdTrackInfo(bbox=bbox, track_id=tid, frame_idx=self._frame_counter)
            self._prev_crowd_tracks.append(info)
            self._crowd_tracks_history.append(info)

        # cleanup history
        max_hist = int(self.cfg.tracking.max_history_frames)
        cutoff = self._frame_counter - max_hist
        self._crowd_tracks_history = [t for t in self._crowd_tracks_history if t.frame_idx >= cutoff]

        # Create mapping from crowd_id to person_ids
        crowd_id_to_person_ids: dict[int, list[int]] = {c["crowd_id"]: c["person_ids"] for c in current_crowds}

        # For each group -> update stack
        extracted: list[ExtractedStack] = []
        for bbox, tid in zip(group_bboxes, group_track_ids):
            stack_id = self._find_or_create_stack(bbox)
            stack = self._movinet_stacks[stack_id]

            local_idx = stack.frame_count
            stack.original_frames.append(frame_bgr.copy())
            stack.frame_count += 1
            stack.last_update_frame = self._frame_counter
            stack.last_frame_crop_bbox = bbox

            cx, cy = bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0
            stack.frame_crowd_centers[local_idx] = (cx, cy)
            stack.frame_crowd_bboxes.append((local_idx, bbox, tid))
            
            # Assign person IDs to this crowd box (from tracking state)
            if self.cfg.tracking.use_bytetrack:
                crowd_person_ids = crowd_id_to_person_ids.get(tid, [])
                stack.frame_crowd_person_ids[local_idx] = crowd_person_ids
            else:
                stack.frame_crowd_person_ids[local_idx] = []

            # Store person boxes for this frame (for large-person full-frame rule)
            # Store as list of tuples (bbox, person_id)
            stack.frame_person_boxes[local_idx] = list(zip(boxes, person_ids))

            if stack.frame_count >= int(self.cfg.stacking.sequence_frames):
                extracted.append(self._finalize_stack(stack))
                # clear & delete stack
                stack.frames.clear()
                stack.original_frames.clear()
                stack.frame_crowd_bboxes.clear()
                stack.frame_crowd_centers.clear()
                stack.frame_crowd_person_ids.clear()
                stack.frame_person_boxes.clear()
                del self._movinet_stacks[stack_id]

        return {
            "violence_stacks": [s for s in extracted if s.label == "violence"],
            "normal_stacks": [s for s in extracted if s.label == "normal"],
            "has_crowd_alert": bool(self._crowd_alert_active),
            "camera_id": camera_id,
            "debug": {
                "persons": persons_debug,
                "crowds": crowds_debug,
            },
        }


def open_video_capture(video_path: str | Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    return cap

