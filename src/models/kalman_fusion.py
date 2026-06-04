from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import numpy as np
from src.models.lifter_3d import COCO17_BONES


@dataclass(frozen=True)
class KalmanConfig:
    dt: float = 1.0 / 30.0
    process_noise: float = 0.025
    measurement_noise: float = 0.09
    min_confidence: float = 0.15
    graph_alpha: float = 0.08
    bone_length_eta: float = 0.35
    graph_iterations: int = 2
    velocity_blend: float = 0.35
    velocity_noise_scale: float = 1.25
    max_process_noise_scale: float = 8.0


class SkeletonKalmanFilter:
    """Constant-velocity Kalman filter with skeleton graph calibration.

    The Kalman step models temporal inertia. A lightweight MAP-style graph step
    then pulls the filtered pose back toward a valid human skeleton topology.
    """

    def __init__(
        self,
        num_keypoints: int = 17,
        config: KalmanConfig | None = None,
        bones: Iterable[tuple[int, int]] = COCO17_BONES,
    ) -> None:
        self.num_keypoints = num_keypoints
        self.config = config or KalmanConfig()
        self.bones = tuple(bones)
        self.dim_pos = num_keypoints * 3
        self.dim_state = self.dim_pos * 2
        self.x = np.zeros((self.dim_state, 1), dtype=np.float64)
        self.p = np.eye(self.dim_state, dtype=np.float64)
        self._laplacian = self._build_laplacian()
        self._graph_solver = self._build_graph_solver()
        self._target_bone_lengths: np.ndarray | None = None
        self._last_measurement: np.ndarray | None = None
        self._last_pose: np.ndarray | None = None
        self.initialized = False

    def update(self, measurement_xyz: np.ndarray, confidence: np.ndarray) -> np.ndarray:
        measurement = np.asarray(measurement_xyz, dtype=np.float64)
        if measurement.shape != (self.num_keypoints, 3):
            raise ValueError(
                f"Expected measurement shaped ({self.num_keypoints}, 3), "
                f"got {measurement.shape}."
            )
        z = measurement.reshape(self.dim_pos, 1)
        conf = np.asarray(confidence, dtype=np.float64).reshape(self.num_keypoints)
        conf = conf.clip(0.0, 1.0)

        if not self.initialized:
            self.x[: self.dim_pos, 0] = z[:, 0]
            self._target_bone_lengths = self._bone_lengths(measurement)
            calibrated = self._apply_graph_constraint(measurement)
            self.x[: self.dim_pos, 0] = calibrated.reshape(-1)
            self._last_measurement = z.copy()
            self._last_pose = calibrated.copy()
            self.initialized = True
            return calibrated.astype(np.float32)

        self._blend_measurement_velocity(z, conf)
        f = self._transition()
        h = self._measurement_matrix()
        q = self._process_noise(self._velocity_process_scale())
        r = self._measurement_noise(conf)

        try:
            x_pred = f @ self.x
            p_pred = f @ self.p @ f.T + q
            innovation = z - h @ x_pred
            s = h @ p_pred @ h.T + r
            k = p_pred @ h.T @ np.linalg.pinv(s)

            self.x = x_pred + k @ innovation
            self.p = (np.eye(self.dim_state) - k @ h) @ p_pred
        except np.linalg.LinAlgError:
            self.x = f @ self.x
        calibrated = self._apply_graph_constraint(
            self.x[: self.dim_pos].reshape(self.num_keypoints, 3)
        )
        if not np.all(np.isfinite(calibrated)):
            calibrated = (
                self._last_pose.copy()
                if self._last_pose is not None
                else measurement.copy()
            )
        self.x[: self.dim_pos, 0] = calibrated.reshape(-1)
        self._last_measurement = z.copy()
        self._last_pose = calibrated.copy()
        return calibrated.astype(np.float32)

    def velocity(self) -> np.ndarray:
        return self.x[self.dim_pos :].reshape(self.num_keypoints, 3).astype(np.float32)

    def set_dt(self, dt: float) -> None:
        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("Kalman dt must be a finite positive value.")
        self.config = KalmanConfig(
            dt=dt,
            process_noise=self.config.process_noise,
            measurement_noise=self.config.measurement_noise,
            min_confidence=self.config.min_confidence,
            graph_alpha=self.config.graph_alpha,
            bone_length_eta=self.config.bone_length_eta,
            graph_iterations=self.config.graph_iterations,
            velocity_blend=self.config.velocity_blend,
            velocity_noise_scale=self.config.velocity_noise_scale,
            max_process_noise_scale=self.config.max_process_noise_scale,
        )

    def reset(self) -> None:
        self.x.fill(0.0)
        self.p = np.eye(self.dim_state, dtype=np.float64)
        self._target_bone_lengths = None
        self._last_measurement = None
        self._last_pose = None
        self.initialized = False

    def _transition(self) -> np.ndarray:
        f = np.eye(self.dim_state, dtype=np.float64)
        f[: self.dim_pos, self.dim_pos :] = np.eye(self.dim_pos) * self.config.dt
        return f

    def _measurement_matrix(self) -> np.ndarray:
        h = np.zeros((self.dim_pos, self.dim_state), dtype=np.float64)
        h[:, : self.dim_pos] = np.eye(self.dim_pos)
        return h

    def _process_noise(self, scale: float = 1.0) -> np.ndarray:
        dt = self.config.dt
        q1 = (dt**4) / 4.0
        q2 = (dt**3) / 2.0
        q3 = dt**2
        block = np.block(
            [
                [np.eye(self.dim_pos) * q1, np.eye(self.dim_pos) * q2],
                [np.eye(self.dim_pos) * q2, np.eye(self.dim_pos) * q3],
            ]
        )
        scale = float(np.clip(scale, 1.0, self.config.max_process_noise_scale))
        return block * self.config.process_noise * scale

    def _velocity_process_scale(self) -> float:
        velocity = self.x[self.dim_pos :].reshape(self.num_keypoints, 3)
        mean_speed = float(np.mean(np.linalg.norm(velocity, axis=1)))
        return 1.0 + self.config.velocity_noise_scale * mean_speed

    def _blend_measurement_velocity(self, z: np.ndarray, confidence: np.ndarray) -> None:
        if self._last_measurement is None:
            return
        dt = max(float(self.config.dt), 1e-6)
        measured_velocity = ((z - self._last_measurement) / dt).reshape(
            self.num_keypoints, 3
        )
        state_velocity = self.x[self.dim_pos :].reshape(self.num_keypoints, 3)
        weights = confidence.clip(0.0, 1.0)[:, None]
        blend = float(np.clip(self.config.velocity_blend, 0.0, 1.0))
        state_velocity[:] = (
            (1.0 - blend * weights) * state_velocity
            + blend * weights * measured_velocity
        )

    def _measurement_noise(self, confidence: np.ndarray) -> np.ndarray:
        confidence = np.maximum(confidence, self.config.min_confidence)
        joint_noise = self.config.measurement_noise / confidence
        repeated = np.repeat(joint_noise, 3)
        return np.diag(repeated)

    def _build_laplacian(self) -> np.ndarray:
        adjacency = np.zeros((self.num_keypoints, self.num_keypoints), dtype=np.float64)
        for a, b in self.bones:
            adjacency[a, b] = 1.0
            adjacency[b, a] = 1.0
        degree = np.diag(adjacency.sum(axis=1))
        return degree - adjacency

    def _build_graph_solver(self) -> np.ndarray:
        laplacian_3d = np.kron(self._laplacian, np.eye(3, dtype=np.float64))
        matrix = np.eye(self.dim_pos, dtype=np.float64) + 2.0 * self.config.graph_alpha * laplacian_3d
        return np.linalg.pinv(matrix)

    def _apply_graph_constraint(self, pose: np.ndarray) -> np.ndarray:
        if self.config.graph_alpha > 0.0:
            pose = (self._graph_solver @ pose.reshape(-1, 1)).reshape(self.num_keypoints, 3)
        if self.config.bone_length_eta > 0.0:
            pose = self._project_bone_lengths(pose)
        return pose

    def _project_bone_lengths(self, pose: np.ndarray) -> np.ndarray:
        if self._target_bone_lengths is None:
            self._target_bone_lengths = self._bone_lengths(pose)

        projected = pose.copy()
        iterations = max(1, int(self.config.graph_iterations))
        eta = float(np.clip(self.config.bone_length_eta, 0.0, 1.0))
        for _ in range(iterations):
            for bone_index, (a, b) in enumerate(self.bones):
                delta = projected[a] - projected[b]
                length = np.linalg.norm(delta)
                if length < 1e-8:
                    continue
                target = self._target_bone_lengths[bone_index]
                corrected_delta = delta * (target / length)
                center = 0.5 * (projected[a] + projected[b])
                corrected_a = center + 0.5 * corrected_delta
                corrected_b = center - 0.5 * corrected_delta
                projected[a] = (1.0 - eta) * projected[a] + eta * corrected_a
                projected[b] = (1.0 - eta) * projected[b] + eta * corrected_b
        return projected

    def _bone_lengths(self, pose: np.ndarray) -> np.ndarray:
        return np.array(
            [np.linalg.norm(pose[a] - pose[b]) for a, b in self.bones],
            dtype=np.float64,
        )
