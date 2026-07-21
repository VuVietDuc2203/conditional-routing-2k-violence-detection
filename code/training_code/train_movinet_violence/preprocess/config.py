from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PathsConfig:
    input_dir: str
    output_dir: str


@dataclass(frozen=True)
class ModelsConfig:
    person_detector_onnx: str
    fight_detector_onnx: str


@dataclass(frozen=True)
class VideoConfig:
    sample_fps: float = 4.0
    extensions: tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv")
    recursive: bool = True
    track_every_frame: bool = True
    track_stride: int = 1


@dataclass(frozen=True)
class PersonDetectionConfig:
    conf_threshold: float = 0.25


@dataclass(frozen=True)
class ClusteringConfig:
    cluster_min_pts: int = 2


@dataclass(frozen=True)
class TrackingConfig:
    overlap_threshold: float = 0.05
    max_history_frames: int = 10
    # ByteTrack settings
    use_bytetrack: bool = True
    tracker: str = "bytetrack"  # "bytetrack" or "botsort"
    track_thresh: float = 0.25  # confidence threshold for tracking
    track_buffer: int = 30  # frames to keep lost tracks
    match_thresh: float = 0.8  # IoU threshold for matching
    min_person_appearances: int = 2  # filter persons appearing < N times in crowd


@dataclass(frozen=True)
class StackingConfig:
    sequence_frames: int = 16
    crop_scale: float = 1.5
    crop_resize_size: int = 172
    large_person_ratio: float = 0.4
    large_person_min_frames: int = 2


@dataclass(frozen=True)
class FightDetectionConfig:
    conf_threshold: float = 0.5
    valid_violence_frames: int = 2


@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "cpu"
    show: bool = False
    save_debug: bool = False
    save_debug_only_crowd: bool = True
    max_cpu_threads: int = 0  # 0 = auto, >0 = limit threads
    batch_fight_detection: bool = True  # batch fight detection for better GPU usage


@dataclass(frozen=True)
class PreprocessConfig:
    paths: PathsConfig
    models: ModelsConfig
    video: VideoConfig = VideoConfig()
    person_detection: PersonDetectionConfig = PersonDetectionConfig()
    clustering: ClusteringConfig = ClusteringConfig()
    tracking: TrackingConfig = TrackingConfig()
    stacking: StackingConfig = StackingConfig()
    fight_detection: FightDetectionConfig = FightDetectionConfig()
    runtime: RuntimeConfig = RuntimeConfig()

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "PreprocessConfig":
        paths = PathsConfig(**d["paths"])
        models = ModelsConfig(**d["models"])
        video = VideoConfig(**d.get("video", {}))
        person_detection = PersonDetectionConfig(**d.get("person_detection", {}))
        clustering = ClusteringConfig(**d.get("clustering", {}))
        tracking = TrackingConfig(**d.get("tracking", {}))
        stacking = StackingConfig(**d.get("stacking", {}))
        fight_detection = FightDetectionConfig(**d.get("fight_detection", {}))
        runtime = RuntimeConfig(**d.get("runtime", {}))
        return PreprocessConfig(
            paths=paths,
            models=models,
            video=video,
            person_detection=person_detection,
            clustering=clustering,
            tracking=tracking,
            stacking=stacking,
            fight_detection=fight_detection,
            runtime=runtime,
        )


def load_config(config_path: str | Path) -> PreprocessConfig:
    config_path = Path(config_path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg = PreprocessConfig.from_dict(raw)

    # Resolve paths relative to repo root / current working directory
    base = config_path.parent
    resolved = PreprocessConfig(
        paths=PathsConfig(
            input_dir=str((base / cfg.paths.input_dir).resolve())
            if not Path(cfg.paths.input_dir).is_absolute()
            else cfg.paths.input_dir,
            output_dir=str((base / cfg.paths.output_dir).resolve())
            if not Path(cfg.paths.output_dir).is_absolute()
            else cfg.paths.output_dir,
        ),
        models=ModelsConfig(
            person_detector_onnx=str((base / cfg.models.person_detector_onnx).resolve())
            if not Path(cfg.models.person_detector_onnx).is_absolute()
            else cfg.models.person_detector_onnx,
            fight_detector_onnx=str((base / cfg.models.fight_detector_onnx).resolve())
            if not Path(cfg.models.fight_detector_onnx).is_absolute()
            else cfg.models.fight_detector_onnx,
        ),
        video=cfg.video,
        person_detection=cfg.person_detection,
        clustering=cfg.clustering,
        tracking=cfg.tracking,
        stacking=cfg.stacking,
        fight_detection=cfg.fight_detection,
        runtime=cfg.runtime,
    )
    return resolved

