from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np


@dataclass(frozen=True)
class SurfaceZone:
    name: str
    kind: str
    polygon: np.ndarray
    height_m: float = 0.0
    rest_allowed: bool = False
    fall_risk: float = 1.0


@dataclass(frozen=True)
class EnvironmentContext:
    zone_name: str | None
    surface_kind: str
    surface_height_m: float
    rest_allowed: bool
    fall_risk: float
    contact_score: float
    topology_score: float
    body_in_rest_region: bool
    body_on_floor: bool

    def vector(self) -> np.ndarray:
        kind_set = ("floor", "bed", "sofa", "tatami", "carpet", "bathroom", "kitchen")
        one_hot = [1.0 if self.surface_kind == kind else 0.0 for kind in kind_set]
        return np.asarray(
            [
                self.surface_height_m,
                float(self.rest_allowed),
                self.fall_risk,
                self.contact_score,
                self.topology_score,
                float(self.body_in_rest_region),
                float(self.body_on_floor),
                *one_hot,
            ],
            dtype=np.float32,
        )


def skeleton_vertical_span(skeleton_xyz: np.ndarray) -> float:
    return float(np.max(skeleton_xyz[:, 1]) - np.min(skeleton_xyz[:, 1]))


class RegionSemanticMap:
    """Polygon semantic map for beds, tatami, carpets, sofas and floor."""

    def __init__(self, zones: list[SurfaceZone]) -> None:
        self.zones = zones

    @classmethod
    def from_json(cls, path: str | Path) -> "RegionSemanticMap":
        config_path = Path(path)
        if not config_path.exists():
            return cls([])
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        zones = [
            SurfaceZone(
                name=item["name"],
                kind=item.get("kind", "floor"),
                polygon=np.asarray(item["polygon"], dtype=np.float32),
                height_m=float(item.get("height_m", 0.0)),
                rest_allowed=bool(item.get("rest_allowed", False)),
                fall_risk=float(item.get("fall_risk", 1.0)),
            )
            for item in raw.get("zones", [])
        ]
        return cls(zones)

    def infer_context(
        self,
        keypoints_xyc: np.ndarray,
        bbox_xyxy: np.ndarray,
        skeleton_xyz: np.ndarray,
    ) -> EnvironmentContext:
        contact_point = self._contact_point(keypoints_xyc, bbox_xyxy)
        zone = self._find_zone(contact_point)
        surface_height_m = 0.0 if zone is None else zone.height_m
        contact_score = self._contact_score(
            keypoints_xyc,
            bbox_xyxy,
            skeleton_xyz,
            surface_height_m,
        )
        topology_score = self._topology_score(zone, skeleton_xyz, contact_score)
        body_in_rest_region = bool(zone is not None and zone.rest_allowed)
        body_on_floor = self._body_on_floor(zone, skeleton_xyz, contact_score)

        if zone is None:
            return EnvironmentContext(
                zone_name=None,
                surface_kind="unknown",
                surface_height_m=surface_height_m,
                rest_allowed=False,
                fall_risk=1.0,
                contact_score=contact_score,
                topology_score=topology_score,
                body_in_rest_region=False,
                body_on_floor=body_on_floor,
            )

        return EnvironmentContext(
            zone_name=zone.name,
            surface_kind=zone.kind,
            surface_height_m=surface_height_m,
            rest_allowed=zone.rest_allowed,
            fall_risk=zone.fall_risk,
            contact_score=contact_score,
            topology_score=topology_score,
            body_in_rest_region=body_in_rest_region,
            body_on_floor=body_on_floor,
        )

    def _find_zone(self, point_xy: np.ndarray) -> SurfaceZone | None:
        zones = sorted(
            self.zones,
            key=lambda item: (item.kind in {"floor", "unknown"}, self._polygon_area(item.polygon)),
        )
        for zone in zones:
            if self._point_in_polygon(point_xy, zone.polygon):
                return zone
        return None

    def _topology_score(
        self, zone: SurfaceZone | None, skeleton_xyz: np.ndarray, contact_score: float
    ) -> float:
        torso_relative_height = self._torso_relative_height(skeleton_xyz)
        horizontal_ratio = self._horizontal_ratio(skeleton_xyz)
        if zone is None:
            return float(np.clip(contact_score * horizontal_ratio, 0.0, 1.0))
        rest_bonus = 0.45 if zone.rest_allowed and torso_relative_height < 0.35 else 0.0
        risk_bonus = 0.25 * max(zone.fall_risk - 1.0, 0.0)
        return float(np.clip(contact_score * horizontal_ratio + risk_bonus - rest_bonus, 0.0, 1.0))

    def _contact_point(self, keypoints_xyc: np.ndarray, bbox_xyxy: np.ndarray) -> np.ndarray:
        ankles = keypoints_xyc[[15, 16]]
        valid_ankles = ankles[ankles[:, 2] > 0.2]
        if len(valid_ankles) > 0:
            return np.mean(valid_ankles[:, :2], axis=0)
        x1, y1, x2, y2 = bbox_xyxy
        return np.array([(x1 + x2) * 0.5, y2], dtype=np.float32)

    @staticmethod
    def _contact_score(
        keypoints_xyc: np.ndarray,
        bbox_xyxy: np.ndarray,
        skeleton_xyz: np.ndarray,
        surface_height_m: float,
    ) -> float:
        del surface_height_m
        x1, y1, x2, y2 = bbox_xyxy
        height = max(float(y2 - y1), 1.0)
        pelvis = np.mean(keypoints_xyc[[11, 12], :2], axis=0)
        ankle_y = np.max(keypoints_xyc[[15, 16], 1])
        image_score = float(np.clip((ankle_y - pelvis[1]) / height, 0.0, 1.0))

        depth_span = float(np.max(skeleton_xyz[:, 2]) - np.min(skeleton_xyz[:, 2]))
        vertical_span = skeleton_vertical_span(skeleton_xyz)
        horizontal_depth_score = np.clip(depth_span / max(vertical_span, 1e-3), 0.0, 1.0)
        score_3d = horizontal_depth_score
        return float(np.clip(0.55 * image_score + 0.45 * score_3d, 0.0, 1.0))

    def _body_on_floor(
        self,
        zone: SurfaceZone | None,
        skeleton_xyz: np.ndarray,
        contact_score: float,
    ) -> bool:
        if zone is None or zone.kind not in {"floor", "carpet"}:
            return False
        horizontal_ratio = self._horizontal_ratio(skeleton_xyz)
        depth_span = float(np.max(skeleton_xyz[:, 2]) - np.min(skeleton_xyz[:, 2]))
        vertical_span = skeleton_vertical_span(skeleton_xyz)
        depth_contact = depth_span >= 0.35 * max(vertical_span, 1e-3)
        return bool(
            contact_score >= 0.45
            and (horizontal_ratio >= 0.45 or depth_contact)
        )

    @staticmethod
    def _horizontal_ratio(skeleton_xyz: np.ndarray) -> float:
        shoulders = np.mean(skeleton_xyz[[5, 6]], axis=0)
        hips = np.mean(skeleton_xyz[[11, 12]], axis=0)
        torso = shoulders - hips
        vertical = abs(float(torso[1]))
        horizontal = float(np.linalg.norm(torso[[0, 2]]))
        return float(horizontal / (horizontal + vertical + 1e-6))

    @staticmethod
    def _torso_relative_height(skeleton_xyz: np.ndarray) -> float:
        span = max(skeleton_vertical_span(skeleton_xyz), 1e-6)
        lowest = float(np.min(skeleton_xyz[:, 1]))
        torso_y = float(np.mean(skeleton_xyz[[5, 6, 11, 12], 1]))
        return float((torso_y - lowest) / span)

    @staticmethod
    def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
        x, y = point
        inside = False
        j = len(polygon) - 1
        for i in range(len(polygon)):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            crosses = (yi > y) != (yj > y)
            if crosses:
                x_on_edge = (xj - xi) * (y - yi) / (yj - yi + 1e-8) + xi
                inside = inside != (x < x_on_edge)
            j = i
        return inside

    @staticmethod
    def _polygon_area(polygon: np.ndarray) -> float:
        if len(polygon) < 3:
            return 0.0
        x = polygon[:, 0]
        y = polygon[:, 1]
        return float(abs(0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))))
