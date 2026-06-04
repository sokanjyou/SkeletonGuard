from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from src.models.yolo26_pose import PoseDetection


@dataclass(frozen=True)
class TrackedDetection:
    track_id: int
    detection: PoseDetection


class SimpleMultiPersonTracker:
    """
    
    Pose-aware tracker for assigning stable IDs to pose detections. ^v^

    
    """

    def __init__(
        self,
        backend: str = "pose_iou",
        iou_threshold: float = 0.25,
        max_missed: int = 15,
        max_center_distance: float = 180.0,
        min_keypoint_confidence: float = 0.2,
        iou_weight: float = 0.45,
        center_weight: float = 0.20,
        keypoint_weight: float = 0.35,
    ) -> None:
        self.backend = backend
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.max_center_distance = max_center_distance
        self.min_keypoint_confidence = min_keypoint_confidence
        weights = np.asarray([iou_weight, center_weight, keypoint_weight], dtype=np.float32)
        weights = weights / max(float(np.sum(weights)), 1e-6)
        self.iou_weight = float(weights[0])
        self.center_weight = float(weights[1])
        self.keypoint_weight = float(weights[2])
        self._next_id = 1
        self._tracks: dict[int, dict[str, np.ndarray | int]] = {}

    def update(self, detections: list[PoseDetection]) -> list[TrackedDetection]:
        for state in self._tracks.values():
            state["missed"] = int(state["missed"]) + 1
            state["age"] = int(state.get("age", 0)) + 1

        assignments: list[TrackedDetection] = []
        used_detections: set[int] = set()
        used_tracks: set[int] = set()

        candidate_pairs = self._candidate_pairs(detections)
        for score, detection_index, track_id in candidate_pairs:
            del score
            if detection_index in used_detections or track_id in used_tracks:
                continue
            detection = detections[detection_index]
            self._update_track(track_id, detection)
            used_detections.add(detection_index)
            used_tracks.add(track_id)
            assignments.append(TrackedDetection(track_id=track_id, detection=detection))

        for detection_index, detection in enumerate(detections):
            if detection_index in used_detections:
                continue
            track_id = self._new_track(detection)
            assignments.append(TrackedDetection(track_id=track_id, detection=detection))

        self._drop_stale_tracks()
        return assignments

    def active_ids(self) -> set[int]:
        return set(self._tracks)

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    def _candidate_pairs(self, detections: list[PoseDetection]) -> list[tuple[float, int, int]]:
        pairs: list[tuple[float, int, int]] = []
        for detection_index, detection in enumerate(detections):
            for track_id, state in self._tracks.items():
                score = self._match_score(detection, state)
                if score is None:
                    continue
                pairs.append((score, detection_index, track_id))
        return sorted(pairs, key=lambda item: item[0], reverse=True)

    def _match_score(
        self,
        detection: PoseDetection,
        state: dict[str, np.ndarray | int],
    ) -> float | None:
        predicted_bbox = self._predicted_bbox(state)
        iou = self._iou(detection.bbox_xyxy, predicted_bbox)
        center_distance = self._center_distance(detection.bbox_xyxy, predicted_bbox)
        center_score = 1.0 - min(center_distance / max(self.max_center_distance, 1.0), 1.0)
        keypoint_score = self._keypoint_similarity(
            detection.keypoints_xyc,
            np.asarray(state["keypoints"], dtype=np.float32),
            detection.bbox_xyxy,
            predicted_bbox,
        )
        if (
            iou < self.iou_threshold
            and center_distance > self.max_center_distance
            and keypoint_score < 0.15
        ):
            return None
        missed_penalty = 0.02 * int(state["missed"])
        score = (
            self.iou_weight * iou
            + self.center_weight * center_score
            + self.keypoint_weight * keypoint_score
            - missed_penalty
        )
        return float(score)

    def _new_track(self, detection: PoseDetection) -> int:
        if detection.track_id is not None and detection.track_id not in self._tracks:
            track_id = int(detection.track_id)
            self._next_id = max(self._next_id, track_id + 1)
        else:
            track_id = self._next_id
            self._next_id += 1
        self._tracks[track_id] = {
            "bbox": detection.bbox_xyxy.copy(),
            "keypoints": detection.keypoints_xyc.copy(),
            "velocity": np.zeros(4, dtype=np.float32),
            "missed": 0,
            "age": 1,
        }
        return track_id

    def _update_track(self, track_id: int, detection: PoseDetection) -> None:
        previous_bbox = np.asarray(self._tracks[track_id]["bbox"], dtype=np.float32)
        velocity = detection.bbox_xyxy.astype(np.float32) - previous_bbox
        self._tracks[track_id] = {
            "bbox": detection.bbox_xyxy.copy(),
            "keypoints": detection.keypoints_xyc.copy(),
            "velocity": velocity,
            "missed": 0,
            "age": int(self._tracks[track_id].get("age", 0)) + 1,
        }

    def _predicted_bbox(self, state: dict[str, np.ndarray | int]) -> np.ndarray:
        bbox = np.asarray(state["bbox"], dtype=np.float32)
        velocity = np.asarray(state.get("velocity", np.zeros(4)), dtype=np.float32)
        missed = max(1, int(state["missed"]))
        return bbox + velocity * min(missed, 3)

    def _drop_stale_tracks(self) -> None:
        stale = [
            track_id
            for track_id, state in self._tracks.items()
            if int(state["missed"]) > self.max_missed
        ]
        for track_id in stale:
            del self._tracks[track_id]

    def _keypoint_similarity(
        self,
        current: np.ndarray,
        previous: np.ndarray,
        current_bbox: np.ndarray,
        previous_bbox: np.ndarray,
    ) -> float:
        current_conf = np.clip(current[:, 2], 0.0, 1.0)
        previous_conf = np.clip(previous[:, 2], 0.0, 1.0)
        weights = np.sqrt(current_conf * previous_conf)
        visible = weights >= self.min_keypoint_confidence
        if not np.any(visible):
            return 0.0
        scale = max(
            self._bbox_scale(current_bbox),
            self._bbox_scale(previous_bbox),
            1.0,
        )
        distances = np.linalg.norm(current[visible, :2] - previous[visible, :2], axis=1)
        visible_weights = weights[visible]
        point_scores = np.exp(-distances / scale)
        weighted_score = float(
            np.sum(point_scores * visible_weights) / max(np.sum(visible_weights), 1e-6)
        )
        coverage = float(np.sum(visible_weights) / max(len(weights), 1))
        return float(np.clip(weighted_score * coverage, 0.0, 1.0))

    @staticmethod
    def _bbox_scale(bbox: np.ndarray) -> float:
        x1, y1, x2, y2 = bbox
        width = max(float(x2 - x1), 1.0)
        height = max(float(y2 - y1), 1.0)
        return float(np.sqrt(width * height))

    def _legacy_match(self, bbox: np.ndarray, used_tracks: set[int]) -> int | None:
        best_id = None
        best_score = -1.0
        for track_id, state in self._tracks.items():
            if track_id in used_tracks:
                continue
            previous_bbox = np.asarray(state["bbox"], dtype=np.float32)
            iou = self._iou(bbox, previous_bbox)
            center_distance = self._center_distance(bbox, previous_bbox)
            if iou < self.iou_threshold and center_distance > self.max_center_distance:
                continue
            score = iou - 0.001 * center_distance
            if score > best_score:
                best_score = score
                best_id = track_id
        return best_id

    @staticmethod
    def _iou(a: np.ndarray, b: np.ndarray) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        intersection = iw * ih
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - intersection
        return float(intersection / union) if union > 1e-8 else 0.0

    @staticmethod
    def _center_distance(a: np.ndarray, b: np.ndarray) -> float:
        ac = np.array([(a[0] + a[2]) * 0.5, (a[1] + a[3]) * 0.5])
        bc = np.array([(b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5])
        return float(np.linalg.norm(ac - bc))
