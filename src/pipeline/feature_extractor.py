from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from src.pipeline.environment_context import EnvironmentContext


@dataclass(frozen=True)
class MotionFeatures:
    vector: np.ndarray
    torso_tilt: float
    hip_height: float
    center_height: float
    pelvis_down_speed: float
    bbox_aspect: float
    confidence_mean: float


@dataclass(frozen=True)
class FeatureExtractorConfig:
    feature_size: int = 32
    min_confidence: float = 0.15
    normalize: bool = True
    normalization_alpha: float = 0.05
    normalization_min_std: float = 1e-3
    normalization_clip: float = 5.0


class SkeletonFeatureExtractor:
    """Builds LSTM-ready per-frame features from pose, motion and context."""

    def __init__(self, config: FeatureExtractorConfig | None = None) -> None:
        self.config = config or FeatureExtractorConfig()
        self.previous_velocity: np.ndarray | None = None
        self.previous_bbox_aspect: float | None = None
        self._feature_mean: np.ndarray | None = None
        self._feature_var: np.ndarray | None = None

    def reset(self) -> None:
        self.previous_velocity = None
        self.previous_bbox_aspect = None
        self._feature_mean = None
        self._feature_var = None

    def extract(
        self,
        skeleton_xyz: np.ndarray,
        velocity_xyz: np.ndarray,
        keypoints_xyc: np.ndarray,
        bbox_xyxy: np.ndarray,
        context: EnvironmentContext,
    ) -> MotionFeatures:
        skeleton = np.asarray(skeleton_xyz, dtype=np.float32)
        velocity = np.asarray(velocity_xyz, dtype=np.float32)
        confidence = np.asarray(keypoints_xyc[:, 2], dtype=np.float32)

        shoulders = np.mean(skeleton[[5, 6]], axis=0)
        hips = np.mean(skeleton[[11, 12]], axis=0)
        head = skeleton[0]
        torso = shoulders - hips
        torso_norm = float(np.linalg.norm(torso) + 1e-6)
        vertical_span = float(np.max(skeleton[:, 1]) - np.min(skeleton[:, 1]))
        horizontal_span = float(np.max(skeleton[:, 0]) - np.min(skeleton[:, 0]))
        depth_span = float(np.max(skeleton[:, 2]) - np.min(skeleton[:, 2]))

        torso_tilt = float(np.linalg.norm(torso[[0, 2]]) / torso_norm)
        head_hip_vertical = float((head[1] - hips[1]) / torso_norm)
        lowest_y = float(np.min(skeleton[:, 1]))
        hip_height = float(hips[1] - lowest_y)
        head_height = float(head[1] - lowest_y)
        center_height = float(np.mean(skeleton[:, 1]) - lowest_y)

        pelvis_velocity = np.mean(velocity[[11, 12]], axis=0)
        pelvis_down_speed = max(0.0, -float(pelvis_velocity[1]))
        joint_speed = np.linalg.norm(velocity, axis=1)
        avg_speed = float(np.mean(joint_speed))
        max_speed = float(np.max(joint_speed))
        pelvis_accel = self._pelvis_acceleration_norm(velocity)

        bbox_aspect = self._bbox_aspect(bbox_xyxy)
        bbox_aspect_rate = 0.0
        if self.previous_bbox_aspect is not None:
            bbox_aspect_rate = bbox_aspect - self.previous_bbox_aspect
        self.previous_bbox_aspect = bbox_aspect

        valid_conf = confidence[confidence >= self.config.min_confidence]
        confidence_mean = float(np.mean(confidence)) if len(confidence) else 0.0
        confidence_min = float(np.min(confidence)) if len(confidence) else 0.0
        low_conf_fraction = float(1.0 - len(valid_conf) / max(len(confidence), 1))

        base = np.asarray(
            [
                torso_tilt,
                head_hip_vertical,
                float(torso[1]),
                hip_height,
                head_height,
                center_height,
                vertical_span,
                horizontal_span,
                depth_span,
                pelvis_down_speed,
                avg_speed,
                max_speed,
                pelvis_accel,
                bbox_aspect,
                bbox_aspect_rate,
                confidence_mean,
                confidence_min,
                low_conf_fraction,
            ],
            dtype=np.float32,
        )
        vector = np.concatenate([base, context.vector()], axis=0)
        vector = self._fit_size(vector)
        vector = self._normalize(vector)
        return MotionFeatures(
            vector=vector,
            torso_tilt=torso_tilt,
            hip_height=hip_height,
            center_height=center_height,
            pelvis_down_speed=pelvis_down_speed,
            bbox_aspect=bbox_aspect,
            confidence_mean=confidence_mean,
        )

    def _pelvis_acceleration_norm(self, velocity: np.ndarray) -> float:
        if self.previous_velocity is None:
            self.previous_velocity = velocity.copy()
            return 0.0
        current = np.mean(velocity[[11, 12]], axis=0)
        previous = np.mean(self.previous_velocity[[11, 12]], axis=0)
        self.previous_velocity = velocity.copy()
        return float(np.linalg.norm(current - previous))

    @staticmethod
    def _bbox_aspect(bbox_xyxy: np.ndarray) -> float:
        x1, y1, x2, y2 = np.asarray(bbox_xyxy, dtype=np.float32)
        width = max(float(x2 - x1), 1.0)
        height = max(float(y2 - y1), 1.0)
        return float(width / height)

    def _fit_size(self, vector: np.ndarray) -> np.ndarray:
        size = max(1, int(self.config.feature_size))
        if len(vector) == size:
            return vector.astype(np.float32)
        if len(vector) > size:
            return vector[:size].astype(np.float32)
        padded = np.zeros(size, dtype=np.float32)
        padded[: len(vector)] = vector
        return padded

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        if not self.config.normalize:
            return vector.astype(np.float32)

        alpha = float(np.clip(self.config.normalization_alpha, 1e-4, 1.0))
        if self._feature_mean is None or self._feature_var is None:
            self._feature_mean = vector.astype(np.float32).copy()
            self._feature_var = np.ones_like(vector, dtype=np.float32)
            return np.zeros_like(vector, dtype=np.float32)

        previous_mean = self._feature_mean.copy()
        self._feature_mean = (1.0 - alpha) * self._feature_mean + alpha * vector
        delta = vector - previous_mean
        self._feature_var = (1.0 - alpha) * self._feature_var + alpha * (delta * delta)
        std = np.sqrt(np.maximum(self._feature_var, self.config.normalization_min_std**2))
        normalized = (vector - self._feature_mean) / std
        clip = float(max(self.config.normalization_clip, 0.0))
        if clip > 0.0:
            normalized = np.clip(normalized, -clip, clip)
        return normalized.astype(np.float32)
