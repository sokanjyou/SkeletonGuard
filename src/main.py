from __future__ import annotations
import argparse
import sys
import time
from collections import deque
from dataclasses import dataclass, replace
from itertools import chain
from pathlib import Path
import cv2
import numpy as np
import yaml
from src.core.stream_loader import VideoStream, ensure_parent
from src.models import (
    KalmanConfig,
    LifterConfig,
    LSTMClassifierConfig,
    LSTMFeatureSequenceClassifier,
    SkeletonKalmanFilter,
    SubspaceSparseLifter,
    YOLOPoseEstimator,
)
from src.pipeline.environment_context import RegionSemanticMap
from src.pipeline.feature_extractor import FeatureExtractorConfig, SkeletonFeatureExtractor
from src.pipeline.fall_detector import FallDecision, FallDetector, FallDetectorConfig
from src.pipeline.multi_person_tracker import SimpleMultiPersonTracker
from src.utils.logger_alerts import build_logger, build_mqtt_alert_publisher, emit_alert
from src.utils.visualizer import draw_pose

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def draw_fps(frame, fps: float | None):
    if fps is None:
        return frame
    cv2.rectangle(frame, (12, 64), (150, 104), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"FPS {fps:.1f}",
        (22, 92),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


@dataclass
class PersonRuntimeState:
    lifter: SubspaceSparseLifter
    kalman: SkeletonKalmanFilter
    feature_extractor: SkeletonFeatureExtractor
    detector: FallDetector
    feature_window: deque[np.ndarray]
    lstm_score_window: deque[float]
    last_seen_frame: int = 0
    last_alert_time: float = 0.0


@dataclass(frozen=True)
class LSTMFusionConfig:
    enabled: bool | str = "auto"
    weights: str | None = None
    device: str | int | None = "auto"
    prefer_cuda: bool = True
    cuda_index: int = 0
    window: int = 18
    score_window: int | None = None
    weight: float = 0.35
    threshold: float | None = None
    sustained_frames: int = 5
    peak_threshold: float = 0.82
    evidence_threshold: float = 0.45


def build_person_state(
    lifter_cfg: LifterConfig,
    kalman_cfg: KalmanConfig,
    feature_cfg: FeatureExtractorConfig,
    fall_cfg: FallDetectorConfig,
    lstm_window: int,
    lstm_score_window: int,
) -> PersonRuntimeState:
    state = PersonRuntimeState(
        lifter=SubspaceSparseLifter(lifter_cfg),
        kalman=SkeletonKalmanFilter(config=kalman_cfg),
        feature_extractor=SkeletonFeatureExtractor(feature_cfg),
        detector=FallDetector(fall_cfg),
        feature_window=deque(maxlen=lstm_window),
        lstm_score_window=deque(maxlen=lstm_score_window),
    )
    state.lifter.reset()
    state.kalman.reset()
    state.feature_extractor.reset()
    state.detector.reset()
    return state


def prune_person_states(
    person_states: dict[int, PersonRuntimeState],
    active_ids: set[int],
    frame_index: int,
    state_ttl_frames: int,
    max_person_states: int,
) -> None:
    stale_ids = [
        track_id
        for track_id, state in person_states.items()
        if track_id not in active_ids
        or frame_index - state.last_seen_frame > state_ttl_frames
    ]
    for track_id in stale_ids:
        del person_states[track_id]

    overflow = len(person_states) - max(0, int(max_person_states))
    if overflow <= 0:
        return
    oldest = sorted(
        person_states,
        key=lambda track_id: person_states[track_id].last_seen_frame,
    )
    for track_id in oldest[:overflow]:
        del person_states[track_id]


def build_lstm_classifier(
    cfg: dict | None,
    logger,
    default_input_size: int,
) -> tuple[LSTMFeatureSequenceClassifier | None, LSTMFusionConfig]:
    cfg = cfg or {}
    fusion_cfg = LSTMFusionConfig(**(cfg.get("fusion") or {}))
    fusion_cfg = replace(fusion_cfg, window=max(1, int(fusion_cfg.window or 1)))
    score_window = fusion_cfg.score_window
    if score_window is None:
        score_window = max(1, int(fusion_cfg.sustained_frames or 1))
    fusion_cfg = replace(fusion_cfg, score_window=max(1, int(score_window)))

    enabled = fusion_cfg.enabled
    auto_enabled = isinstance(enabled, str) and enabled.lower() == "auto"
    if enabled is False:
        return None, fusion_cfg

    if not fusion_cfg.weights:
        logger.warning(
            "LSTM fusion is %s but no weights were configured; skipping LSTM branch.",
            "auto" if auto_enabled else "enabled",
        )
        return None, fusion_cfg

    weights_path = Path(fusion_cfg.weights)
    if not weights_path.is_absolute():
        weights_path = PROJECT_ROOT / weights_path
    if not weights_path.exists():
        message = "LSTM weights not found at %s; skipping LSTM branch."
        if auto_enabled:
            logger.info(message, weights_path)
        else:
            logger.warning(message, weights_path)
        return None, fusion_cfg

    model_options = cfg.get("model") or {}
    if model_options.get("input_size") is None:
        model_options = {**model_options, "input_size": default_input_size}
    model_cfg = LSTMClassifierConfig(**model_options)
    classifier = LSTMFeatureSequenceClassifier(
        weights_path=str(weights_path),
        device=fusion_cfg.device,
        prefer_cuda=fusion_cfg.prefer_cuda,
        cuda_index=fusion_cfg.cuda_index,
        config=model_cfg,
    )
    logger.info(
            "Loaded LSTM feature classifier from %s on device=%s input_size=%s",
            weights_path,
            classifier.device,
            classifier.expected_input_size,
    )
    return classifier, fusion_cfg


def frame_dt_for_stream(
    stream: VideoStream,
    frame_started: float,
    previous_frame_started: float | None,
    fallback_dt: float,
) -> float:
    if stream.is_live_source and previous_frame_started is not None:
        return float(np.clip(frame_started - previous_frame_started, 1.0 / 240.0, 1.0))
    return fallback_dt


def fuse_lstm_decision(
    decision: FallDecision,
    lstm_classifier: LSTMFeatureSequenceClassifier | None,
    feature_window: deque[np.ndarray],
    lstm_score_window: deque[float],
    fusion_cfg: LSTMFusionConfig,
    fallback_threshold: float,
) -> FallDecision:
    if lstm_classifier is None or len(feature_window) < fusion_cfg.window:
        return decision

    sequence = np.stack(list(feature_window), axis=0)
    lstm_score = lstm_classifier.predict_proba(sequence)
    lstm_weight = float(np.clip(fusion_cfg.weight, 0.0, 1.0))
    fused_score = float(
        np.clip(
            (1.0 - lstm_weight) * decision.score + lstm_weight * lstm_score,
            0.0,
            1.0,
        )
    )
    threshold = fusion_cfg.threshold if fusion_cfg.threshold is not None else fallback_threshold
    lstm_score_window.append(fused_score)
    stable_score = float(np.mean(lstm_score_window))
    recent = list(lstm_score_window)[-max(1, int(fusion_cfg.sustained_frames)) :]
    sustained = (
        len(recent) >= max(1, int(fusion_cfg.sustained_frames))
        and all(score >= threshold for score in recent)
        and stable_score >= threshold * 0.85
    )
    peak_score = max(lstm_score_window) if lstm_score_window else fused_score
    peak_evidence = (
        peak_score >= fusion_cfg.peak_threshold
        and decision.impact_score >= fusion_cfg.evidence_threshold
        and decision.posture_score >= fusion_cfg.evidence_threshold
        and decision.height_score >= 0.35
    )
    reason = (
        f"{decision.reason}, lstm={lstm_score:.2f}, "
        f"fused={stable_score:.2f}"
    )
    return replace(
        decision,
        is_fall=bool(decision.is_fall or sustained or peak_evidence),
        score=stable_score,
        reason=reason,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Monocular 3D fall detection")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--source", default=None)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    logger = build_logger()
    stream = None
    writer = None
    alert_publisher = None

    try:
        cfg = load_yaml(args.config)
        source = args.source if args.source is not None else cfg.get("source", 0)

        pose_model = YOLOPoseEstimator(**cfg.get("pose", {}))
        logger.info("YOLO26-Pose inference device=%s", pose_model.device)
        lifter_cfg = LifterConfig(**cfg.get("lifter", {}))
        kalman_cfg = KalmanConfig(**cfg.get("kalman", {}))
        feature_cfg = FeatureExtractorConfig(**cfg.get("features", {}))
        fall_cfg = FallDetectorConfig(**cfg.get("fall", {}))
        alert_publisher = build_mqtt_alert_publisher(cfg.get("alerts", {}), logger)
        lstm_classifier, lstm_fusion_cfg = build_lstm_classifier(
            cfg.get("lstm", {}),
            logger,
            default_input_size=feature_cfg.feature_size,
        )
        semantic_map = RegionSemanticMap.from_json(cfg.get("zones", "config/zones_config.json"))
        tracker = SimpleMultiPersonTracker(**cfg.get("tracker", {}))
        tracker.reset()
        person_states: dict[int, PersonRuntimeState] = {}
        runtime_cfg = cfg.get("runtime") or {}
        state_ttl_frames = int(
            runtime_cfg.get("state_ttl_frames", max(30, tracker.max_missed * 2))
        )
        max_person_states = int(runtime_cfg.get("max_person_states", 128))
        alerts_cfg = cfg.get("alerts") or {}
        mqtt_alerts_cfg = alerts_cfg.get("mqtt") or {}
        alert_min_interval_s = float(
            alerts_cfg.get("min_interval_s", mqtt_alerts_cfg.get("min_interval_s", 2.0))
        )

        stream = VideoStream(source, **(cfg.get("stream") or {}))
        source_fps = stream.fps
        source_frame_dt = 1.0 / max(source_fps, 1e-3)
        logger.info("Video source fps=%.2f, initial Kalman dt=%.4fs", source_fps, source_frame_dt)
        if args.output:
            ensure_parent(args.output)
            first_frame = next(stream)
            height, width = first_frame.shape[:2]
            fourcc = cv2.VideoWriter.fourcc(*"mp4v")
            writer = cv2.VideoWriter(args.output, fourcc, stream.fps, (width, height))
            frames = [first_frame]
        else:
            frames = []

        iterator = chain(frames, stream) if frames else stream
        fps_ema = None
        previous_frame_started = None
        for frame_index, frame in enumerate(iterator, start=1):
            frame_started = time.perf_counter()
            current_dt = frame_dt_for_stream(
                stream,
                frame_started,
                previous_frame_started,
                source_frame_dt,
            )
            previous_frame_started = frame_started
            detections = pose_model.infer(frame)
            annotated = frame.copy()
            if detections:
                tracked_people = tracker.update(detections)
                prune_person_states(
                    person_states,
                    tracker.active_ids(),
                    frame_index,
                    state_ttl_frames,
                    max_person_states,
                )

                for tracked in tracked_people:
                    if tracked.track_id not in person_states:
                        person_states[tracked.track_id] = build_person_state(
                            lifter_cfg,
                            kalman_cfg,
                            feature_cfg,
                            fall_cfg,
                            lstm_fusion_cfg.window,
                            lstm_fusion_cfg.score_window or 1,
                        )
                    state = person_states[tracked.track_id]
                    state.last_seen_frame = frame_index
                    person = tracked.detection
                    pose3d = state.lifter.lift(person.keypoints_xyc, frame.shape[:2])
                    state.kalman.set_dt(current_dt)
                    smooth3d = state.kalman.update(pose3d, person.keypoints_xyc[:, 2])
                    velocity3d = state.kalman.velocity()
                    context = semantic_map.infer_context(
                        person.keypoints_xyc,
                        person.bbox_xyxy,
                        smooth3d,
                    )
                    features = state.feature_extractor.extract(
                        smooth3d,
                        velocity3d,
                        person.keypoints_xyc,
                        person.bbox_xyxy,
                        context,
                    )
                    state.feature_window.append(features.vector.copy())
                    decision = state.detector.decide(
                        smooth3d,
                        velocity3d,
                        context,
                        features=features,
                    )
                    decision = fuse_lstm_decision(
                        decision,
                        lstm_classifier,
                        state.feature_window,
                        state.lstm_score_window,
                        lstm_fusion_cfg,
                        fall_cfg.threshold,
                    )
                    annotated = draw_pose(
                        annotated,
                        person.keypoints_xyc,
                        decision,
                        track_id=tracked.track_id,
                        bbox_xyxy=person.bbox_xyxy,
                    )
                    now = time.time()
                    if (
                        decision.is_fall
                        and now - state.last_alert_time >= alert_min_interval_s
                    ):
                        state.last_alert_time = now
                        emit_alert(
                            logger,
                            f"id={tracked.track_id}, {decision.reason}",
                            publisher=alert_publisher,
                            payload={
                                "track_id": tracked.track_id,
                                "score": decision.score,
                                "raw_score": decision.raw_score,
                                "posture_score": decision.posture_score,
                                "impact_score": decision.impact_score,
                                "height_score": decision.height_score,
                                "context_factor": decision.context_factor,
                                "reason": decision.reason,
                            },
                        )
            else:
                tracker.update([])
                prune_person_states(
                    person_states,
                    tracker.active_ids(),
                    frame_index,
                    state_ttl_frames,
                    max_person_states,
                )

            elapsed = max(time.perf_counter() - frame_started, 1e-6)
            instant_fps = 1.0 / elapsed
            fps_ema = instant_fps if fps_ema is None else 0.9 * fps_ema + 0.1 * instant_fps
            annotated = draw_fps(annotated, fps_ema)

            if writer is not None:
                writer.write(annotated)
            if args.show:
                cv2.imshow("fall3d", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except StopIteration:
        logger.info("Video source ended before any frame was read.")
    except Exception as exc:
        logger.exception("Fall detection pipeline failed: %s", exc)
    finally:
        if stream is not None:
            stream.release()
        if writer is not None:
            writer.release()
        if alert_publisher is not None:
            alert_publisher.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
