from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
from src.utils.device import resolve_yolo_device


@dataclass(frozen=True)
class PoseDetection:
    keypoints_xyc: np.ndarray
    bbox_xyxy: np.ndarray
    confidence: float
    track_id: int | None = None


class YOLOPoseEstimator:
    """Thin Ultralytics wrapper that defaults to YOLO26-Pose weights."""

    def __init__(
        self,
        weights: str = "data/models/yolo26m-pose.pt",
        device: str | int | None = "auto",
        prefer_cuda: bool = True,
        cuda_index: int = 0,
        conf: float = 0.35,
        iou: float = 0.5,
        require_pose: bool = True,
    ) -> None:
        self.project_root = Path(__file__).resolve().parents[2]
        os.environ.setdefault("YOLO_CONFIG_DIR", str(self.project_root))

        from ultralytics import YOLO

        self.weights = self._resolve_weights(weights)
        self.model = YOLO(self.weights)
        self._validate_task(require_pose=require_pose)
        self.device = resolve_yolo_device(
            device=device,
            prefer_cuda=prefer_cuda,
            cuda_index=cuda_index,
        )
        self.conf = conf
        self.iou = iou

    def _resolve_weights(self, weights: str) -> str:
        path = Path(weights)
        if path.exists():
            return str(path)

        project_root = self.project_root
        local_path = project_root / weights
        if local_path.exists():
            return str(local_path)

        bundled_yolo26 = project_root / "data" / "models" / "yolo26m-pose.pt"
        if bundled_yolo26.exists():
            return str(bundled_yolo26)

        return weights

    def _validate_task(self, require_pose: bool) -> None:
        task = getattr(self.model, "task", None)
        if require_pose and task != "pose":
            raise ValueError(
                "YOLOPoseEstimator requires a pose model with keypoint output. "
                f"Loaded weights '{self.weights}' are task='{task}'. "
                "Please place a YOLO26 pose weight such as yolo26m-pose.pt under "
                "data/models and update config/config.yaml."
            )

    def infer(self, frame: np.ndarray) -> list[PoseDetection]:
        results = self.model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []
        return self._parse_result(results[0])

    def _parse_result(self, result: Any) -> list[PoseDetection]:
        if result.keypoints is None or result.boxes is None:
            return []

        xy = result.keypoints.xy.cpu().numpy()
        conf = result.keypoints.conf.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()
        box_conf = result.boxes.conf.cpu().numpy()
        ids = None
        if getattr(result.boxes, "id", None) is not None:
            ids = result.boxes.id.cpu().numpy().astype(int)

        detections: list[PoseDetection] = []
        for index in range(len(xy)):
            keypoints_xyc = np.concatenate([xy[index], conf[index, :, None]], axis=1)
            detections.append(
                PoseDetection(
                    keypoints_xyc=keypoints_xyc.astype(np.float32),
                    bbox_xyxy=boxes[index].astype(np.float32),
                    confidence=float(box_conf[index]),
                    track_id=None if ids is None else int(ids[index]),
                )
            )
        return detections
