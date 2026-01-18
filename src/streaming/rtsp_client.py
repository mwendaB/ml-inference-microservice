import threading
import time
from typing import Optional

from src.core.inference_engine import InferenceEngine
from src.monitoring.metrics import MetricsRecorder
from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import cv2
except Exception:  # pragma: no cover - optional dependency
    cv2 = None


class RTSPProcessor:
    def __init__(
        self,
        stream_url: str,
        model: str,
        backend: str,
        fps_limit: int = 30,
        save_output: bool = False,
        output_path: str = "output.mp4",
    ) -> None:
        self.stream_url = stream_url
        self.fps_limit = fps_limit
        self.save_output = save_output
        self.output_path = output_path
        self._engine = InferenceEngine(model, backend, device="auto")
        self._metrics = MetricsRecorder()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def _run(self) -> None:
        if cv2 is None:
            logger.error("opencv_required_for_rtsp")
            return
        writer = None
        while self._running:
            cap = cv2.VideoCapture(self.stream_url)
            if not cap.isOpened():
                logger.warning("rtsp_connect_failed")
                time.sleep(2)
                continue
            if self.save_output:
                fps = cap.get(cv2.CAP_PROP_FPS) or self.fps_limit
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                writer = cv2.VideoWriter(
                    self.output_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )
            frame_interval = 1.0 / max(self.fps_limit, 1)
            last_frame_time = 0.0
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("rtsp_frame_failed")
                    break
                now = time.time()
                if now - last_frame_time < frame_interval:
                    continue
                last_frame_time = now
                result = self._engine.infer(frame)
                self._metrics.record_inference(result["inference_time_ms"], result["model_used"], result["backend"])
                if writer is not None:
                    writer.write(frame)
            cap.release()
            if writer is not None:
                writer.release()
            time.sleep(1)
