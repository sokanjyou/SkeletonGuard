from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Iterable
import numpy as np
from src.pipeline.environment_context import EnvironmentContext
from src.pipeline.environment_context import skeleton_vertical_span
from src.pipeline.feature_extractor import MotionFeatures


@dataclass(frozen=True)
class FallDecision:
    is_fall: bool
    score: float
    raw_score: float
    posture_score: float
    impact_score: float
    height_score: float
    context_factor: float
    reason: str


@dataclass(frozen=True)
class FallDetectorConfig:
    threshold: float = 0.68
    window: int = 18
    sustained_frames: int = 5
    peak_threshold: float = 0.82
    low_height_m: float = 0.45
    impact_velocity: float = 1.15
    rest_suppression: float = 0.55
    risk_boost: float = 0.25
    weight_posture: float = 0.42
    weight_impact: float = 0.28
    weight_height: float = 0.30
    rest_contact_threshold: float = 0.35
    rest_impact_threshold: float = 0.35
    high_risk_surface_kinds: tuple[str, ...] = ("stairs", "bathroom", "kitchen")
    base_context_factor: float = 1.0
    fall_risk_baseline: float = 1.0
    fall_risk_weight: float = 0.20
    topology_weight: float = 0.15
    strong_evidence_raw_threshold: float = 0.82
    strong_evidence_impact_threshold: float = 0.55
    context_factor_min: float = 0.25
    context_factor_max: float = 1.45
    stable_score_threshold_ratio: float = 0.85
    fast_fall_impact_threshold: float = 0.55
    fast_fall_posture_threshold: float = 0.55
    fast_fall_height_threshold: float = 0.45
    omnidirectional_impact_weight: float = 0.75
    min_expected_upright_span: float = 0.75
    upright_span_low_height_multiplier: float = 3.2

    def __post_init__(self) -> None:
        object.__setattr__(self, "window", self._coerce_positive_int("window", self.window))
        object.__setattr__(
            self,
            "sustained_frames",
            self._coerce_positive_int("sustained_frames", self.sustained_frames),
        )
        self._validate_positive("impact_velocity", self.impact_velocity)
        self._validate_positive("low_height_m", self.low_height_m)
        self._validate_positive("base_context_factor", self.base_context_factor)
        self._validate_positive("min_expected_upright_span", self.min_expected_upright_span)
        self._validate_positive(
            "upright_span_low_height_multiplier",
            self.upright_span_low_height_multiplier,
        )

        for name in (
            "threshold",
            "peak_threshold",
            "rest_contact_threshold",
            "rest_impact_threshold",
            "strong_evidence_raw_threshold",
            "strong_evidence_impact_threshold",
            "stable_score_threshold_ratio",
            "fast_fall_impact_threshold",
            "fast_fall_posture_threshold",
            "fast_fall_height_threshold",
            "omnidirectional_impact_weight",
        ):
            self._validate_unit_interval(name, getattr(self, name))

        for name in (
            "rest_suppression",
            "risk_boost",
            "fall_risk_baseline",
            "fall_risk_weight",
            "topology_weight",
        ):
            self._validate_non_negative(name, getattr(self, name))

        if not np.isfinite(float(self.context_factor_min)) or self.context_factor_min <= 0.0:
            raise ValueError("fall.context_factor_min must be > 0.")
        if not np.isfinite(float(self.context_factor_max)):
            raise ValueError("fall.context_factor_max must be finite.")
        if self.context_factor_max < self.context_factor_min:
            raise ValueError("fall.context_factor_max must be >= fall.context_factor_min.")

        weights = np.asarray(
            [self.weight_posture, self.weight_impact, self.weight_height],
            dtype=np.float64,
        )
        if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("fall score weights must be finite non-negative values.")
        weight_sum = float(np.sum(weights))
        if weight_sum <= 0.0:
            raise ValueError("at least one fall score weight must be > 0.")
        normalized = weights / weight_sum
        object.__setattr__(self, "weight_posture", float(normalized[0]))
        object.__setattr__(self, "weight_impact", float(normalized[1]))
        object.__setattr__(self, "weight_height", float(normalized[2]))
        object.__setattr__(
            self,
            "high_risk_surface_kinds",
            self._coerce_kind_tuple(self.high_risk_surface_kinds),
        )

    @staticmethod
    def _validate_positive(name: str, value: float) -> None:
        if not np.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"fall.{name} must be a finite value > 0.")

    @staticmethod
    def _validate_non_negative(name: str, value: float) -> None:
        if not np.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"fall.{name} must be a finite value >= 0.")

    @staticmethod
    def _validate_unit_interval(name: str, value: float) -> None:
        if not np.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"fall.{name} must be in [0, 1].")

    @staticmethod
    def _coerce_positive_int(name: str, value: int | float) -> int:
        numeric = float(value)
        if not np.isfinite(numeric) or numeric <= 0.0 or not numeric.is_integer():
            raise ValueError(f"fall.{name} must be a positive integer.")
        return int(numeric)

    @staticmethod
    def _coerce_kind_tuple(value: Iterable[str]) -> tuple[str, ...]:
        kinds = tuple(str(item).strip() for item in value if str(item).strip())
        if not kinds:
            raise ValueError("fall.high_risk_surface_kinds must not be empty.")
        return kinds


class FallDetector:
    def __init__(self, config: FallDetectorConfig | None = None) -> None:
        self.config = config or FallDetectorConfig()
        self.score_history: deque[float] = deque(maxlen=self.config.window)
        self.raw_history: deque[float] = deque(maxlen=self.config.window)
        self.impact_history: deque[float] = deque(maxlen=self.config.window)
        self._consecutive_risky_frames = 0

    def reset(self) -> None:
        self.score_history.clear()
        self.raw_history.clear()
        self.impact_history.clear()
        self._consecutive_risky_frames = 0

    def decide(
        self,
        skeleton_xyz: np.ndarray,
        velocity_xyz: np.ndarray,
        context: EnvironmentContext,
        features: MotionFeatures | None = None,
    ) -> FallDecision:
        posture_score = features.torso_tilt if features is not None else self._posture_score(skeleton_xyz)
        height_score = self._height_score(skeleton_xyz)
        if context.body_on_floor:
            height_score = max(height_score, context.contact_score)
        impact_score = self._impact_score(velocity_xyz)
        if features is not None:
            impact_score = max(
                impact_score,
                float(np.clip(features.pelvis_down_speed / self.config.impact_velocity, 0.0, 1.0)),
            )
        raw_score = (
            self.config.weight_posture * posture_score
            + self.config.weight_impact * impact_score
            + self.config.weight_height * height_score
        )
        context_factor = self._context_factor(context, raw_score, impact_score)
        score = float(np.clip(raw_score * context_factor, 0.0, 1.0))
        self.score_history.append(score)
        self.raw_history.append(float(raw_score))
        self.impact_history.append(float(impact_score))
        stable_score = float(np.mean(self.score_history))
        if score >= self.config.threshold:
            self._consecutive_risky_frames += 1
        else:
            self._consecutive_risky_frames = 0
        is_fall = self._is_alarm(stable_score, posture_score, impact_score, height_score)
        reason = self._reason(context, posture_score, impact_score, height_score, stable_score)
        return FallDecision(
            is_fall=is_fall,
            score=stable_score,
            raw_score=float(raw_score),
            posture_score=float(posture_score),
            impact_score=float(impact_score),
            height_score=float(height_score),
            context_factor=float(context_factor),
            reason=reason,
        )

    def _posture_score(self, skeleton_xyz: np.ndarray) -> float:
        shoulders = np.mean(skeleton_xyz[[5, 6]], axis=0)
        hips = np.mean(skeleton_xyz[[11, 12]], axis=0)
        torso = shoulders - hips
        vertical = abs(float(torso[1]))
        horizontal = float(np.linalg.norm(torso[[0, 2]]))
        return float(horizontal / (horizontal + vertical + 1e-6))

    def _height_score(self, skeleton_xyz: np.ndarray) -> float:
        vertical_span = skeleton_vertical_span(skeleton_xyz)
        expected_upright_span = max(
            self.config.min_expected_upright_span,
            self.config.low_height_m * self.config.upright_span_low_height_multiplier,
        )
        return float(np.clip(1.0 - vertical_span / expected_upright_span, 0.0, 1.0))

    def _impact_score(self, velocity_xyz: np.ndarray) -> float:
        pelvis_velocity = np.mean(velocity_xyz[[11, 12]], axis=0)
        downward_speed = max(0.0, -float(pelvis_velocity[1]))
        total_speed = float(np.linalg.norm(pelvis_velocity))
        impact_speed = max(
            downward_speed,
            total_speed * self.config.omnidirectional_impact_weight,
        )
        return float(np.clip(impact_speed / self.config.impact_velocity, 0.0, 1.0))

    def _context_factor(
        self, context: EnvironmentContext, raw_score: float, impact_score: float
    ) -> float:
        factor = self.config.base_context_factor
        if (
            context.body_in_rest_region
            and context.contact_score > self.config.rest_contact_threshold
            and impact_score < self.config.rest_impact_threshold
        ):
            factor -= self.config.rest_suppression
        if context.surface_kind in self.config.high_risk_surface_kinds or context.body_on_floor:
            factor += self.config.risk_boost
        factor += self.config.fall_risk_weight * max(
            context.fall_risk - self.config.fall_risk_baseline,
            0.0,
        )
        factor += self.config.topology_weight * context.topology_score
        if (
            raw_score > self.config.strong_evidence_raw_threshold
            and impact_score > self.config.strong_evidence_impact_threshold
        ):
            factor = max(factor, self.config.base_context_factor)
        return float(
            np.clip(
                factor,
                self.config.context_factor_min,
                self.config.context_factor_max,
            )
        )

    def _is_alarm(
        self,
        stable_score: float,
        posture_score: float,
        impact_score: float,
        height_score: float,
    ) -> bool:
        sustained = (
            self._consecutive_risky_frames >= max(1, int(self.config.sustained_frames))
            and stable_score >= self.config.threshold * self.config.stable_score_threshold_ratio
        )
        peak_score = max(self.score_history) if self.score_history else 0.0
        recent_impact = max(self.impact_history) if self.impact_history else impact_score
        fast_fall_pattern = (
            peak_score >= self.config.peak_threshold
            and recent_impact >= self.config.fast_fall_impact_threshold
            and posture_score >= self.config.fast_fall_posture_threshold
            and height_score >= self.config.fast_fall_height_threshold
        )
        return bool(sustained or fast_fall_pattern)

    @staticmethod
    def _reason(
        context: EnvironmentContext,
        posture_score: float,
        impact_score: float,
        height_score: float,
        score: float,
    ) -> str:
        zone = context.zone_name or context.surface_kind
        return (
            f"zone={zone}, posture={posture_score:.2f}, "
            f"impact={impact_score:.2f}, height={height_score:.2f}, score={score:.2f}"
        )
