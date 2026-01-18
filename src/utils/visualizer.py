from typing import List, Tuple

from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import cv2
except Exception:  # pragma: no cover - optional dependency
    cv2 = None


def draw_detections(frame, detections: List[dict]) -> Tuple[object, List[dict]]:
    if cv2 is None:
        return frame, detections
    for det in detections:
        x1, y1, x2, y2 = [int(x) for x in det["bbox"]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        label = f"{det['class_name']} {det['confidence']:.2f}"
        cv2.putText(frame, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    return frame, detections
