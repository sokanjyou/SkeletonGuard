from __future__ import annotations
import cv2
import numpy as np
from src.models.lifter_3d import COCO17_BONES
from src.pipeline.fall_detector import FallDecision


def draw_pose(
    frame: np.ndarray,
    keypoints_xyc: np.ndarray,
    decision: FallDecision | None = None,
    track_id: int | None = None,
    bbox_xyxy: np.ndarray | None = None,
) -> np.ndarray:
    canvas = frame.copy()
    color = (0, 0, 255) if decision and decision.is_fall else (0, 210, 80)
    if bbox_xyxy is not None:
        x1, y1, x2, y2 = bbox_xyxy.astype(int)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

    for a, b in COCO17_BONES:
        if keypoints_xyc[a, 2] < 0.2 or keypoints_xyc[b, 2] < 0.2:
            continue
        pa = tuple(keypoints_xyc[a, :2].astype(int))
        pb = tuple(keypoints_xyc[b, :2].astype(int))
        cv2.line(canvas, pa, pb, color, 2, cv2.LINE_AA)
    for x, y, conf in keypoints_xyc:
        if conf < 0.2:
            continue
        cv2.circle(canvas, (int(x), int(y)), 3, (255, 255, 255), -1, cv2.LINE_AA)

    if decision is not None:
        label = "FALL" if decision.is_fall else "NORMAL"
        prefix = f"ID {track_id} " if track_id is not None else ""
        text = f"{prefix}{label} {decision.score:.2f}"
        if bbox_xyxy is not None:
            x1, y1, _, _ = bbox_xyxy.astype(int)
            left, top = max(0, x1), max(0, y1 - 42)
        else:
            left, top = 12, 12
        cv2.rectangle(canvas, (left, top), (left + 220, top + 38), (0, 0, 0), -1)
        cv2.putText(
            canvas,
            text,
            (left + 10, top + 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )
    return canvas
