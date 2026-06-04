from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import numpy as np


COCO17_BONES: tuple[tuple[int, int], ...] = (
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)


@dataclass(frozen=True)
class LifterConfig:
    num_keypoints: int = 17
    subspace_dim: int = 18
    max_iterations: int = 12
    min_iterations: int = 3
    convergence_tol: float = 1e-4
    iterations: int | None = None
    step_size: float = 0.035
    l1_lambda: float = 0.015
    coefficient_smooth_weight: float = 0.08
    reprojection_weight: float = 1.0
    bone_weight: float = 0.35
    depth_smooth_weight: float = 0.08
    min_confidence: float = 0.15
    camera_scale: float = 2.0
    fallback_on_nonconvergence: bool = True

    @property
    def iteration_limit(self) -> int:
        value = self.iterations if self.iterations is not None else self.max_iterations
        return max(1, int(value))


class SubspaceSparseLifter:
    """
    
    Lifts 2D COCO-17 keypoints into a constrained 3D skeleton. ^v^

    """

    def __init__(
        self,
        config: LifterConfig | None = None,
        bones: Iterable[tuple[int, int]] = COCO17_BONES,
        basis: np.ndarray | None = None,
    ) -> None:
        self.config = config or LifterConfig()
        self.bones = tuple(bones)
        self.mean_pose = self._canonical_pose()
        self.basis = basis if basis is not None else self._make_basis()
        self.target_bone_lengths = self._bone_lengths(self.mean_pose)
        self.previous_pose: np.ndarray | None = None
        self.previous_coefficients: np.ndarray | None = None
        self.last_iterations = 0
        self.last_converged = False

    def lift(self, keypoints_xyc: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:

        points = np.asarray(keypoints_xyc, dtype=np.float64)
        if points.shape != (self.config.num_keypoints, 3):
            raise ValueError(
                "Expected keypoints shaped "
                f"({self.config.num_keypoints}, 3), got {points.shape}."
            )

        xy = points[:, :2]
        conf = points[:, 2].clip(0.0, 1.0)
        normalized_xy = self._normalize_2d(xy, image_shape, conf)

        alpha = np.zeros(self.config.subspace_dim, dtype=np.float64)
        pose = self.mean_pose.copy()
        high_confidence = conf >= self.config.min_confidence
        pose[high_confidence, :2] = normalized_xy[high_confidence]
        pose = self._repair_low_confidence_points(pose, conf)
        alpha = self.basis.T @ (pose.reshape(-1) - self.mean_pose.reshape(-1))
        if self.previous_coefficients is not None:
            alpha = 0.65 * alpha + 0.35 * self.previous_coefficients

        self.last_iterations = 0
        converged = False
        for iteration in range(self.config.iteration_limit):
            pose = (self.mean_pose.reshape(-1) + self.basis @ alpha).reshape(
                self.config.num_keypoints, 3
            )
            grad_pose = self._objective_gradient(pose, normalized_xy, conf)
            grad_alpha = self.basis.T @ grad_pose.reshape(-1)
            if self.previous_coefficients is not None:
                grad_alpha += 2.0 * self.config.coefficient_smooth_weight * (
                    alpha - self.previous_coefficients
                )
            next_alpha = self._soft_threshold(
                alpha - self.config.step_size * grad_alpha,
                self.config.step_size * self.config.l1_lambda,
            )
            self.last_iterations = iteration + 1
            if self._has_converged(alpha, next_alpha, iteration):
                alpha = next_alpha
                converged = True
                break
            alpha = next_alpha

        self.last_converged = converged
        if (
            not converged
            and self.config.fallback_on_nonconvergence
            and self.previous_pose is not None
        ):
            return self.previous_pose.astype(np.float32)

        lifted = (self.mean_pose.reshape(-1) + self.basis @ alpha).reshape(
            self.config.num_keypoints, 3
        )
        lifted = self._project_bone_lengths(lifted)
        lifted = self._root_center(lifted)
        self.previous_pose = lifted.copy()
        self.previous_coefficients = alpha.copy()
        return lifted.astype(np.float32)

    def reset(self) -> None:
        self.previous_pose = None
        self.previous_coefficients = None
        self.last_iterations = 0
        self.last_converged = False

    def _objective_gradient(
        self, pose: np.ndarray, target_xy: np.ndarray, confidence: np.ndarray
    ) -> np.ndarray:
        grad = np.zeros_like(pose)
        valid = np.where(
            confidence >= self.config.min_confidence,
            confidence.clip(0.0, 1.0),
            0.0,
        )
        grad[:, :2] += (
            self.config.reprojection_weight
            * valid[:, None]
            * (pose[:, :2] - target_xy)
        )

        for bone_index, (a, b) in enumerate(self.bones):
            delta = pose[a] - pose[b]
            length = np.linalg.norm(delta) + 1e-8
            error = length - self.target_bone_lengths[bone_index]
            direction = delta / length
            grad[a] += self.config.bone_weight * error * direction
            grad[b] -= self.config.bone_weight * error * direction

        if self.previous_pose is not None:
            grad[:, 2] += self.config.depth_smooth_weight * (
                pose[:, 2] - self.previous_pose[:, 2]
            )
        return grad

    def _project_bone_lengths(self, pose: np.ndarray) -> np.ndarray:
        projected = pose.copy()
        for bone_index, (a, b) in enumerate(self.bones):
            center = 0.5 * (projected[a] + projected[b])
            delta = projected[a] - projected[b]
            length = np.linalg.norm(delta)
            if length < 1e-8:
                continue
            half = 0.5 * self.target_bone_lengths[bone_index] * delta / length
            projected[a] = center + half
            projected[b] = center - half
        return projected

    def _normalize_2d(
        self,
        xy: np.ndarray,
        image_shape: tuple[int, int],
        confidence: np.ndarray,
    ) -> np.ndarray:
        height, width = image_shape[:2]
        scale = max(width, height) / self.config.camera_scale
        normalized = xy.copy()
        normalized[:, 0] = (normalized[:, 0] - width / 2.0) / scale
        normalized[:, 1] = -(normalized[:, 1] - height / 2.0) / scale
        root = self._estimate_2d_root(normalized, confidence)
        normalized -= root
        return normalized

    def _estimate_2d_root(self, normalized_xy: np.ndarray, confidence: np.ndarray) -> np.ndarray:
        hips = (11, 12)
        visible_hips = [index for index in hips if confidence[index] >= self.config.min_confidence]
        if visible_hips:
            return np.mean(normalized_xy[visible_hips], axis=0)

        visible = confidence >= self.config.min_confidence
        if np.any(visible):
            return np.mean(normalized_xy[visible], axis=0)
        return np.zeros(2, dtype=np.float64)

    def _repair_low_confidence_points(
        self, pose: np.ndarray, confidence: np.ndarray
    ) -> np.ndarray:
        repaired = pose.copy()
        source = self.previous_pose if self.previous_pose is not None else self.mean_pose
        low = confidence < self.config.min_confidence
        repaired[low] = source[low]
        return repaired

    def _canonical_pose(self) -> np.ndarray:
        pose = np.zeros((self.config.num_keypoints, 3), dtype=np.float64)
        pose[0] = [0.00, 0.92, 0.00]
        pose[5] = [-0.22, 0.55, 0.00]
        pose[6] = [0.22, 0.55, 0.00]
        pose[7] = [-0.38, 0.28, 0.02]
        pose[8] = [0.38, 0.28, 0.02]
        pose[9] = [-0.42, 0.02, 0.04]
        pose[10] = [0.42, 0.02, 0.04]
        pose[11] = [-0.16, 0.00, 0.00]
        pose[12] = [0.16, 0.00, 0.00]
        pose[13] = [-0.16, -0.46, 0.03]
        pose[14] = [0.16, -0.46, 0.03]
        pose[15] = [-0.16, -0.92, 0.05]
        pose[16] = [0.16, -0.92, 0.05]
        pose[1] = [-0.05, 0.96, 0.00]
        pose[2] = [0.05, 0.96, 0.00]
        pose[3] = [-0.10, 0.90, 0.00]
        pose[4] = [0.10, 0.90, 0.00]
        return self._root_center(pose)

    def _make_basis(self) -> np.ndarray:
        raw = np.zeros(
            (self.config.num_keypoints * 3, self.config.subspace_dim),
            dtype=np.float64,
        )

        def add_mode(index: int, values: dict[int, tuple[float, float, float]]) -> None:
            if index >= raw.shape[1]:
                return
            for joint, delta in values.items():
                raw[joint * 3 : joint * 3 + 3, index] = delta

        y = self.mean_pose[:, 1]
        x = self.mean_pose[:, 0]
        z = self.mean_pose[:, 2]
        raw[1::3, 0] = y
        raw[0::3, 1] = x
        raw[2::3, 2] = y + 0.25 * z
        raw[0::3, 3] = y
        raw[2::3, 4] = y
        add_mode(5, {5: (-0.08, 0.02, 0.0), 7: (-0.24, 0.02, 0.04), 9: (-0.34, 0.0, 0.08)})
        add_mode(6, {6: (0.08, 0.02, 0.0), 8: (0.24, 0.02, 0.04), 10: (0.34, 0.0, 0.08)})
        add_mode(7, {11: (-0.03, 0.0, 0.0), 13: (-0.12, -0.08, 0.06), 15: (-0.16, -0.12, 0.12)})
        add_mode(8, {12: (0.03, 0.0, 0.0), 14: (0.12, -0.08, 0.06), 16: (0.16, -0.12, 0.12)})
        add_mode(9, {7: (-0.06, 0.14, 0.0), 9: (-0.08, 0.28, 0.0)})
        add_mode(10, {8: (0.06, 0.14, 0.0), 10: (0.08, 0.28, 0.0)})
        add_mode(11, {13: (0.02, 0.22, 0.0), 15: (0.02, 0.34, 0.0)})
        add_mode(12, {14: (-0.02, 0.22, 0.0), 16: (-0.02, 0.34, 0.0)})
        add_mode(13, {0: (0.0, 0.08, 0.0), 1: (-0.02, 0.08, 0.0), 2: (0.02, 0.08, 0.0)})
        add_mode(14, {5: (-0.08, 0.0, 0.0), 6: (0.08, 0.0, 0.0)})
        add_mode(15, {11: (-0.06, 0.0, 0.0), 12: (0.06, 0.0, 0.0)})
        add_mode(16, {13: (0.0, 0.24, 0.0), 14: (0.0, 0.24, 0.0), 15: (0.0, 0.28, 0.0), 16: (0.0, 0.28, 0.0)})
        add_mode(17, {15: (-0.12, 0.0, 0.04), 16: (0.12, 0.0, 0.04)})

        if self.config.subspace_dim > 18:
            rng = np.random.default_rng(7)
            raw[:, 18:] = rng.normal(0.0, 0.05, raw[:, 18:].shape)
        q, _ = np.linalg.qr(raw)
        return q[:, : self.config.subspace_dim]

    def _bone_lengths(self, pose: np.ndarray) -> np.ndarray:
        return np.array(
            [np.linalg.norm(pose[a] - pose[b]) for a, b in self.bones],
            dtype=np.float64,
        )

    @staticmethod
    def _soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
        return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)

    def _has_converged(
        self, alpha: np.ndarray, next_alpha: np.ndarray, iteration: int
    ) -> bool:
        if iteration + 1 < self.config.min_iterations:
            return False
        delta = np.linalg.norm(next_alpha - alpha)
        scale = np.linalg.norm(alpha) + 1e-8
        return bool(delta / scale < self.config.convergence_tol)

    @staticmethod
    def _root_center(pose: np.ndarray) -> np.ndarray:
        root = 0.5 * (pose[11] + pose[12])
        return pose - root


class Lifter3D(SubspaceSparseLifter):
    """

    SubspaceSparseLifter. ^v^

    """

    def __init__(
        self,
        model_path: str | None = None,
        device: str | None = None,
        config: LifterConfig | None = None,
    ) -> None:
        del model_path, device
        super().__init__(config=config)

    def predict(self, kpts_2d: np.ndarray, image_shape: tuple[int, int] = (720, 1280)) -> np.ndarray:
        points = np.asarray(kpts_2d, dtype=np.float32)
        if points.shape[-1] == 2:
            confidence = np.ones((points.shape[0], 1), dtype=np.float32)
            points = np.concatenate([points, confidence], axis=1)
        return self.lift(points, image_shape)
