import os
import tempfile
from typing import Optional

from src.core.inference_engine import InferenceEngine
from src.core.tracker import ByteTrack
from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import cv2
except Exception:  # pragma: no cover - optional dependency
    cv2 = None


class VideoProcessor:
    def __init__(self, model: str, backend: str) -> None:
        self._engine = InferenceEngine(model, backend, device="auto")
        self._tracker = ByteTrack()

    def process(self, input_path: str, output_path: Optional[str] = None) -> str:
        if cv2 is None:
            raise RuntimeError("opencv is required for video processing")
        if output_path is None:
            output_path = os.path.join(tempfile.gettempdir(), "tracked_output.mp4")

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError("Unable to open video")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            result = self._engine.infer(frame)
            tracks = self._tracker.update(result["detections"])
            for track in tracks:
                x1, y1, x2, y2 = [int(x) for x in track.bbox]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"ID {track.track_id}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )
            writer.write(frame)

        cap.release()
        writer.release()
        return output_path
