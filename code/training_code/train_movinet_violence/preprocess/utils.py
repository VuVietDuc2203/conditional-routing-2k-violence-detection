from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

BBox = tuple[int, int, int, int]  # (x, y, w, h)


def clamp_bbox_xywh(b: BBox, frame_w: int, frame_h: int) -> BBox:
    x, y, w, h = b
    if w <= 0 or h <= 0:
        return (0, 0, 0, 0)

    x = max(0, min(x, frame_w - 1))
    y = max(0, min(y, frame_h - 1))
    w = max(1, min(w, frame_w - x))
    h = max(1, min(h, frame_h - y))
    return (x, y, w, h)


def xyxy_to_xywh(x1: float, y1: float, x2: float, y2: float) -> BBox:
    x = int(round(min(x1, x2)))
    y = int(round(min(y1, y2)))
    w = int(round(abs(x2 - x1)))
    h = int(round(abs(y2 - y1)))
    return (x, y, w, h)


def area_xywh(b: BBox) -> int:
    _, _, w, h = b
    return max(0, w) * max(0, h)


def overlap_ratio(b1: BBox, b2: BBox) -> float:
    """intersection_area / min(area1, area2) in [0, 1]."""
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0:
        return 0.0

    ax1, ay1, ax2, ay2 = x1, y1, x1 + w1, y1 + h1
    bx1, by1, bx2, by2 = x2, y2, x2 + w2, y2 + h2

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    min_area = min(area_xywh(b1), area_xywh(b2))
    if min_area <= 0:
        return 0.0
    return float(inter) / float(min_area)


def union_bbox(b1: BBox, b2: BBox) -> BBox:
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    ax1, ay1, ax2, ay2 = x1, y1, x1 + w1, y1 + h1
    bx1, by1, bx2, by2 = x2, y2, x2 + w2, y2 + h2
    ux1, uy1 = min(ax1, bx1), min(ay1, by1)
    ux2, uy2 = max(ax2, bx2), max(ay2, by2)
    return (ux1, uy1, max(0, ux2 - ux1), max(0, uy2 - uy1))


def union_bboxes(boxes: Iterable[BBox]) -> BBox:
    it = iter(boxes)
    try:
        u = next(it)
    except StopIteration:
        return (0, 0, 0, 0)
    for b in it:
        u = union_bbox(u, b)
    return u


def expand_bbox(b: BBox, scale: float, frame_w: int, frame_h: int) -> BBox:
    x, y, w, h = b
    if w <= 0 or h <= 0:
        return (0, 0, 0, 0)

    cx = x + w / 2.0
    cy = y + h / 2.0
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    nx = int(round(cx - nw / 2.0))
    ny = int(round(cy - nh / 2.0))
    return clamp_bbox_xywh((nx, ny, nw, nh), frame_w, frame_h)


def crop_and_resize(frame_bgr: np.ndarray, bbox: BBox, size: int) -> np.ndarray:
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        # fallback: center crop square of min side
        h0, w0 = frame_bgr.shape[:2]
        side = min(h0, w0)
        x = (w0 - side) // 2
        y = (h0 - side) // 2
        w = h = side
    crop = frame_bgr[y : y + h, x : x + w]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)


def list_videos(input_dir: str | Path, extensions: tuple[str, ...], recursive: bool) -> list[Path]:
    input_dir = Path(input_dir)
    if not input_dir.exists():
        return []
    exts = {e.lower() for e in extensions}
    if recursive:
        candidates = input_dir.rglob("*")
    else:
        candidates = input_dir.glob("*")
    out: list[Path] = []
    for p in candidates:
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p)
    out.sort()
    return out

