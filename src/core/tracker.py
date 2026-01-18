import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

from src.utils.config_loader import ConfigLoader
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Track:
    track_id: int
    bbox: List[float]
    class_id: int
    confidence: float
    last_seen: float
    first_seen: float
    hits: int = 1
    age: int = 0
    history: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=50))
    inside_zones: Dict[str, bool] = field(default_factory=dict)

    def update(self, bbox: List[float], confidence: float, timestamp: float) -> None:
        self.bbox = bbox
        self.confidence = confidence
        self.last_seen = timestamp
        self.hits += 1
        self.age = 0
        self.history.append(self.center())

    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def dwell_time(self) -> float:
        return self.last_seen - self.first_seen


class ByteTrack:
    def __init__(self, config_path: str = "config/tracking.yaml") -> None:
        self._config_loader = ConfigLoader(config_path)
        self._config = self._config_loader.get().get("tracking", {})
        self._tracks: Dict[int, Track] = {}
        self._next_id = 1
        self._entry_count: Dict[str, int] = {}
        self._exit_count: Dict[str, int] = {}

    def update(self, detections: List[Dict[str, float]]) -> List[Track]:
        timestamp = time.time()
        matches = self._match_tracks(detections)

        used_detection_indices = set()
        for track_id, det_idx in matches:
            detection = detections[det_idx]
            self._tracks[track_id].update(detection["bbox"], detection["confidence"], timestamp)
            used_detection_indices.add(det_idx)

        for idx, det in enumerate(detections):
            if idx in used_detection_indices:
                continue
            if det["confidence"] < self._config.get("track_thresh", 0.5):
                continue
            self._add_track(det, timestamp)

        self._age_tracks()
        self._remove_stale_tracks()
        self._update_zone_counts()
        return list(self._tracks.values())

    def get_counts(self) -> Dict[str, Dict[str, int]]:
        return {"entries": self._entry_count, "exits": self._exit_count}

    def _add_track(self, detection: Dict[str, float], timestamp: float) -> None:
        track = Track(
            track_id=self._next_id,
            bbox=detection["bbox"],
            class_id=detection.get("class_id", 0),
            confidence=detection.get("confidence", 0.0),
            last_seen=timestamp,
            first_seen=timestamp,
        )
        track.history.append(track.center())
        self._tracks[self._next_id] = track
        self._next_id += 1

    def _age_tracks(self) -> None:
        for track in self._tracks.values():
            track.age += 1

    def _remove_stale_tracks(self) -> None:
        max_age = self._config.get("max_age", 30)
        stale = [track_id for track_id, track in self._tracks.items() if track.age > max_age]
        for track_id in stale:
            self._tracks.pop(track_id, None)

    def _match_tracks(self, detections: List[Dict[str, float]]) -> List[Tuple[int, int]]:
        matches: List[Tuple[int, int]] = []
        for track_id, track in self._tracks.items():
            best_iou = 0.0
            best_idx = -1
            for idx, det in enumerate(detections):
                if det["confidence"] < self._config.get("track_thresh", 0.5):
                    continue
                iou_val = _iou(track.bbox, det["bbox"])
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_idx = idx
            if best_iou >= self._config.get("match_thresh", 0.8) and best_idx >= 0:
                matches.append((track_id, best_idx))
        return matches

    def _update_zone_counts(self) -> None:
        zones = self._config.get("zones", [])
        for zone in zones:
            name = zone.get("name", "zone")
            polygon = zone.get("polygon", [])
            if not polygon:
                continue
            for track in self._tracks.values():
                inside = _point_in_polygon(track.center(), polygon)
                prev_inside = track.inside_zones.get(name, False)
                if inside and not prev_inside:
                    self._entry_count[name] = self._entry_count.get(name, 0) + 1
                if not inside and prev_inside:
                    self._exit_count[name] = self._exit_count.get(name, 0) + 1
                track.inside_zones[name] = inside


def _iou(box_a: List[float], box_b: List[float]) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _point_in_polygon(point: Tuple[float, float], polygon: List[List[float]]) -> bool:
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        intersects = ((y1 > y) != (y2 > y)) and (
            x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-6) + x1
        )
        if intersects:
            inside = not inside
    return inside
