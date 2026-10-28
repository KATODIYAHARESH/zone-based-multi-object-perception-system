"""Typed, validated configuration: ``config.yaml`` -> :class:`AppConfig`.

* Unknown keys are rejected (``extra="forbid"``) so that a typo such as
  ``confidense: 0.4`` fails loudly at start-up instead of silently running on a
  default - a classic field failure mode in long-running deployments.
* Runtime mutation is validated too (``validate_assignment=True``), so the
  dashboard cannot push an out-of-range value into a live engine.
* Ethics invariants are encoded in the *type system*: ``store_face_images`` is
  ``Literal[False]`` and ``monitoring_indicator`` is ``Literal[True]``.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z0-9_]+)(?::-([^}]*))?\}")
SeverityName = Literal["info", "warning", "critical"]
FaceBackend = Literal["insightface", "yunet", "mediapipe"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class SystemConfig(_Strict):
    name: str = "ZBMOPS-01"
    site: str = "default-site"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_dir: str = "data/logs"
    data_dir: str = "data"


class ReconnectConfig(_Strict):
    enabled: bool = True
    initial_delay_s: float = Field(0.5, gt=0)
    max_delay_s: float = Field(10.0, gt=0)
    max_attempts: int = Field(0, ge=0)


class SourceConfig(_Strict):
    uri: str = "0"
    width: int = Field(1280, ge=160)
    height: int = Field(720, ge=120)
    resize: bool = True
    fps_hint: float = Field(30.0, gt=0, le=240)
    loop_video: bool = True
    realtime_playback: bool = True
    read_timeout_s: float = Field(5.0, gt=0)
    reconnect: ReconnectConfig = Field(default_factory=ReconnectConfig)


class PerformanceConfig(_Strict):
    device: str = "auto"
    half_precision: bool = True
    imgsz: int = Field(640, ge=160, le=1920)
    detect_every_n_frames: int = Field(1, ge=1, le=30)
    adaptive_skip: bool = True
    target_fps: float = Field(25.0, gt=0, le=240)
    max_skip: int = Field(4, ge=1, le=30)
    warmup: bool = True


class OpenVocabConfig(_Strict):
    enabled: bool = True
    backend: Literal["yolo_world", "yoloe", "grounding_dino"] = "yolo_world"
    model: str = "yolov8s-worldv2.pt"
    confidence: float = Field(0.25, ge=0.0, le=1.0)
    interval_frames: int = Field(15, ge=1)
    merge_iou: float = Field(0.5, ge=0.0, le=1.0)
    prompts: Dict[str, List[str]] = Field(default_factory=dict)


class MotionConfig(_Strict):
    enabled: bool = True
    only_in_zones: bool = True
    min_area_frac: float = Field(0.002, gt=0, lt=1)
    max_area_frac: float = Field(0.25, gt=0, le=1)
    history: int = Field(300, ge=10)
    var_threshold: float = Field(32.0, gt=0)
    covered_frac: float = Field(0.4, ge=0, le=1)
    confidence: float = Field(0.65, ge=0, le=1)
    downscale: float = Field(0.5, gt=0, le=1)
    persist_frames: int = Field(5, ge=1)


class PlateConfig(_Strict):
    enabled: bool = True
    model: Optional[str] = None
    confidence: float = Field(0.3, ge=0, le=1)
    min_vehicle_height_px: int = Field(80, ge=10)


class DetectionConfig(_Strict):
    model: str = "yolo26s.pt"
    confidence: float = Field(0.35, ge=0.01, le=1.0)
    iou: float = Field(0.5, ge=0.05, le=0.95)
    category_nms_iou: float = Field(0.7, ge=0.05, le=1.0)
    unclassified_floor: float = Field(0.15, ge=0.0, le=1.0)
    max_det: int = Field(300, ge=1)
    agnostic_nms: bool = False
    classes: Optional[List[int]] = None
    open_vocab: OpenVocabConfig = Field(default_factory=OpenVocabConfig)
    motion: MotionConfig = Field(default_factory=MotionConfig)
    plates: PlateConfig = Field(default_factory=PlateConfig)

    @model_validator(mode="after")
    def _floor_below_confidence(self) -> "DetectionConfig":
        if self.unclassified_floor > self.confidence:
            raise ValueError("detection.unclassified_floor must be <= detection.confidence")
        return self


class TrackingConfig(_Strict):
    high_thresh: Optional[float] = Field(None, ge=0, le=1)
    low_thresh: Optional[float] = Field(None, ge=0, le=1)
    new_track_thresh: float = Field(0.45, ge=0, le=1)
    match_thresh: float = Field(0.8, gt=0, le=1)
    second_match_thresh: float = Field(0.5, gt=0, le=1)
    track_buffer: int = Field(30, ge=1)
    min_hits: int = Field(3, ge=1)
    class_aware: bool = True
    label_consensus: float = Field(0.6, ge=0, le=1)
    vote_decay: float = Field(0.9, gt=0, le=1)


class EmotionConfig(_Strict):
    enabled: bool = True
    backend: Literal["hsemotion", "deepface", "none"] = "hsemotion"
    model: str = "enet_b0_8_best_vgaf"
    smoothing: float = Field(0.6, ge=0, lt=1)


class AgeConfig(_Strict):
    enabled: bool = True
    groups: List[Tuple[int, int, str]] = Field(default_factory=lambda: [
        (0, 12, "child"), (13, 19, "teen"), (20, 35, "20-35"), (36, 55, "36-55"), (56, 120, "56+")])


class RecognitionConfig(_Strict):
    enabled: bool = True
    threshold: float = Field(0.45, ge=0, le=1)
    gallery_db: str = "data/faces.db"


class FaceConfig(_Strict):
    enabled: bool = True
    detector: FaceBackend = "insightface"
    fallback_order: List[FaceBackend] = Field(default_factory=lambda: ["insightface", "yunet", "mediapipe"])
    insightface_model: str = "buffalo_l"
    det_size: int = Field(320, ge=128, le=1280)
    yunet_model: str = "models/face_detection_yunet_2023mar.onnx"
    arcface_model: Optional[str] = None
    min_face_px: int = Field(28, ge=8)
    min_det_score: float = Field(0.5, ge=0, le=1)
    every_n_frames: int = Field(6, ge=1)
    max_faces_per_cycle: int = Field(8, ge=1)
    emotion: EmotionConfig = Field(default_factory=EmotionConfig)
    age: AgeConfig = Field(default_factory=AgeConfig)
    recognition: RecognitionConfig = Field(default_factory=RecognitionConfig)


class ZonesConfig(_Strict):
    file: str = "zones.yaml"
    enter_frames: int = Field(3, ge=1)
    exit_frames: int = Field(5, ge=1)
    anchor: Literal["bottom_center", "center"] = "bottom_center"


class _Channel(_Strict):
    enabled: bool = False
    min_severity: SeverityName = "info"
    max_per_minute: int = Field(60, ge=1)


class OverlayChannel(_Channel):
    enabled: bool = True
    max_items: int = Field(50, ge=1)


class LogFileChannel(_Channel):
    enabled: bool = True
    path: str = "data/logs/alerts.jsonl"


class TelegramChannel(_Channel):
    min_severity: SeverityName = "warning"
    bot_token_env: str = "TELEGRAM_BOT_TOKEN"
    chat_id_env: str = "TELEGRAM_CHAT_ID"
    send_snapshot: bool = True


class EmailChannel(_Channel):
    min_severity: SeverityName = "critical"
    smtp_host_env: str = "SMTP_HOST"
    smtp_port: int = Field(587, ge=1, le=65535)
    username_env: str = "SMTP_USER"
    password_env: str = "SMTP_PASSWORD"
    sender: str = "zbmops@localhost"
    recipients: List[str] = Field(default_factory=list)
    use_tls: bool = True


class WebhookChannel(_Channel):
    url: str = "http://localhost:3000/api/events"
    heartbeat_url: Optional[str] = "http://localhost:3000/api/heartbeat"
    token_env: str = "INGEST_TOKEN"
    forward_zone_events: bool = True
    timeout_s: float = Field(3.0, gt=0)


class ChannelsConfig(_Strict):
    overlay: OverlayChannel = Field(default_factory=OverlayChannel)
    log_file: LogFileChannel = Field(default_factory=LogFileChannel)
    telegram: TelegramChannel = Field(default_factory=TelegramChannel)
    email: EmailChannel = Field(default_factory=EmailChannel)
    webhook: WebhookChannel = Field(default_factory=WebhookChannel)


class AlertsConfig(_Strict):
    default_cooldown_s: float = Field(30.0, ge=0)
    snapshot_on_alert: bool = True
    rules: List[Dict[str, Any]] = Field(default_factory=list)
    channels: ChannelsConfig = Field(default_factory=ChannelsConfig)


class HeatmapConfig(_Strict):
    enabled: bool = True
    cell_px: int = Field(8, ge=1)
    decay: float = Field(0.997, gt=0, le=1)
    alpha: float = Field(0.45, ge=0, le=1)


class AnalyticsConfig(_Strict):
    db_path: str = "data/events.db"
    snapshots_dir: str = "data/snapshots"
    export_dir: str = "data/exports"
    counts_interval_s: float = Field(5.0, gt=0)
    heatmap: HeatmapConfig = Field(default_factory=HeatmapConfig)


class PrivacyConfig(_Strict):
    blur_faces: bool = False
    blur_enrolled: bool = False
    blur_snapshots: bool = True
    store_face_images: Literal[False] = False      # cannot be enabled - by design
    retention_days: int = Field(30, ge=1, le=3650)
    purge_interval_s: float = Field(3600, ge=60)
    monitoring_indicator: Literal[True] = True     # cannot be disabled - by design
    indicator_text: str = "MONITORING ACTIVE"


class StreamServerConfig(_Strict):
    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = Field(8765, ge=1, le=65535)
    jpeg_quality: int = Field(80, ge=10, le=100)
    max_fps: float = Field(25.0, gt=0)


class UIConfig(_Strict):
    mode: Literal["window", "headless"] = "window"
    window_name: str = "ZBMOPS // Mission Control"
    show_heatmap: bool = False
    show_hud: bool = True
    stream_server: StreamServerConfig = Field(default_factory=StreamServerConfig)


class AppConfig(_Strict):
    system: SystemConfig = Field(default_factory=SystemConfig)
    source: SourceConfig = Field(default_factory=SourceConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    face: FaceConfig = Field(default_factory=FaceConfig)
    zones: ZonesConfig = Field(default_factory=ZonesConfig)
    alerts: AlertsConfig = Field(default_factory=AlertsConfig)
    analytics: AnalyticsConfig = Field(default_factory=AnalyticsConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    ui: UIConfig = Field(default_factory=UIConfig)

    _base_dir: Path = PrivateAttr(default_factory=Path.cwd)

    @model_validator(mode="after")
    def _cross_checks(self) -> "AppConfig":
        ov = self.detection.open_vocab
        if ov.enabled and ov.interval_frames >= self.tracking.track_buffer:
            raise ValueError("detection.open_vocab.interval_frames must be < tracking.track_buffer, "
                             "otherwise open-vocabulary-only tracks expire between passes")
        return self

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    def path(self, value: "str | os.PathLike[str]") -> Path:
        """Resolve a config path relative to the config file's directory."""
        p = Path(value).expanduser()
        return p if p.is_absolute() else (self._base_dir / p).resolve()


def _expand_env(node: Any) -> Any:
    """Expand ``${VAR}`` / ``${VAR:-default}`` in every string value."""
    if isinstance(node, dict):
        return {k: _expand_env(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_expand_env(v) for v in node]
    if isinstance(node, str):
        return _ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1), m.group(2) or ""), node)
    return node


def load_config(path: "str | os.PathLike[str]" = "config.yaml") -> AppConfig:
    """Load, env-expand and validate ``config.yaml``.

    Raises:
        FileNotFoundError: the file does not exist.
        pydantic.ValidationError: a value is invalid or a key is unknown.
    """
    import yaml  # local import keeps pure-logic modules importable without PyYAML

    cfg_path = Path(path).expanduser().resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{cfg_path}: top level must be a mapping")
    cfg = AppConfig.model_validate(_expand_env(raw))
    cfg._base_dir = cfg_path.parent
    return cfg
