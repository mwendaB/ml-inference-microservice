import asyncio
import io
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.core.model_manager import ModelManager
from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import cv2
except Exception:  # pragma: no cover - optional dependency
    cv2 = None

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency
    Image = None

try:
    from ultralytics.engine.results import Results
except Exception:  # pragma: no cover - optional dependency
    Results = None


class InferenceEngine:
    def __init__(
        self,
        model_name: str,
        backend: str,
        device: str,
        batch_size: int = 1,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        class_filter: Optional[List[int]] = None,
    ) -> None:
        self.model_name = model_name
        self.backend = backend
        self.device = device
        self.batch_size = batch_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_filter = class_filter
        self._manager = ModelManager()
        self._metrics: Dict[str, Any] = {
            "latencies_ms": deque(maxlen=200),
            "requests": 0,
        }

    async def infer_async(self, image: Any) -> Dict[str, Any]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.infer, image)

    def infer(self, image: Any) -> Dict[str, Any]:
        start = time.perf_counter()
        handle = self._manager.get_model(self.model_name, self.backend, self.device)
        detections, shape = self._run_inference(handle.model, image)
        duration_ms = (time.perf_counter() - start) * 1000
        self._metrics["latencies_ms"].append(duration_ms)
        self._metrics["requests"] += 1
        return {
            "detections": detections,
            "inference_time_ms": duration_ms,
            "image_shape": shape,
            "model_used": handle.model_name,
            "backend": handle.backend,
        }

    def infer_batch(self, images: List[Any]) -> List[Dict[str, Any]]:
        return [self.infer(image) for image in images]

    def get_metrics(self) -> Dict[str, Any]:
        latencies = list(self._metrics["latencies_ms"])
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        return {
            "avg_latency_ms": avg_latency,
            "requests": self._metrics["requests"],
        }

    def _run_inference(self, model: Any, image: Any) -> Tuple[List[Dict[str, Any]], Tuple[int, int]]:
        if isinstance(image, np.ndarray):
            frame = image
        else:
            frame = self._to_numpy(image)
        height, width = frame.shape[:2]

        if hasattr(model, "predict"):
            results = model.predict(frame, conf=self.conf_threshold, iou=self.iou_threshold)
            return self._parse_results(results), (height, width)
        if isinstance(model, list):
            return self._parse_results(model), (height, width)
        raise RuntimeError("Unsupported model inference method")

    def _to_numpy(self, image: Any) -> np.ndarray:
        if cv2 is not None and isinstance(image, bytes):
            data = np.frombuffer(image, dtype=np.uint8)
            frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if frame is not None:
                return frame
        if Image is not None and isinstance(image, bytes):
            pil = Image.open(io.BytesIO(image)).convert("RGB")
            return np.array(pil)
        if hasattr(image, "read"):
            data = image.read()
            if cv2 is not None:
                frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    return frame
            if Image is not None:
                pil = Image.open(io.BytesIO(data)).convert("RGB")
                return np.array(pil)
        raise ValueError("Unsupported image input")

    def _parse_results(self, results: Any) -> List[Dict[str, Any]]:
        detections: List[Dict[str, Any]] = []
        if Results is not None and results and isinstance(results[0], Results):
            for result in results:
                names = result.names
                boxes = result.boxes
                if boxes is None:
                    continue
                for box in boxes:
                    class_id = int(box.cls)
                    if self.class_filter and class_id not in self.class_filter:
                        continue
                    confidence = float(box.conf)
                    if confidence < self.conf_threshold:
                        continue
                    x1, y1, x2, y2 = [float(coord) for coord in box.xyxy[0].tolist()]
                    detections.append(
                        {
                            "class_id": class_id,
                            "class_name": names.get(class_id, str(class_id)),
                            "confidence": confidence,
                            "bbox": [x1, y1, x2, y2],
                        }
                    )
            return detections

        if isinstance(results, list) and results and isinstance(results[0], dict):
            payload = results[0]
            for bbox, score, class_id, class_name in zip(
                payload.get("boxes", []),
                payload.get("scores", []),
                payload.get("class_ids", []),
                payload.get("class_names", []),
            ):
                if self.class_filter and class_id not in self.class_filter:
                    continue
                if score < self.conf_threshold:
                    continue
                detections.append(
                    {
                        "class_id": int(class_id),
                        "class_name": str(class_name),
                        "confidence": float(score),
                        "bbox": [float(x) for x in bbox],
                    }
                )
        return detections
