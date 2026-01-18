import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from src.utils.config_loader import ConfigLoader
from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - optional dependency
    YOLO = None

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover - optional dependency
    ort = None

try:
    import tensorrt as trt
except Exception:  # pragma: no cover - optional dependency
    trt = None


@dataclass
class ModelHandle:
    model_name: str
    backend: str
    device: str
    model: Any


class DummyModel:
    def predict(self, images: Any, conf: float = 0.25, iou: float = 0.45) -> list[dict]:
        return [
            {
                "boxes": [[10.0, 10.0, 100.0, 100.0]],
                "scores": [0.9],
                "class_ids": [0],
                "class_names": ["object"],
            }
        ]


class ModelManager:
    def __init__(self, config_path: str = "config/models.yaml") -> None:
        self._lock = threading.RLock()
        self._config = ConfigLoader(config_path)
        self._models: Dict[Tuple[str, str, str], ModelHandle] = {}
        self._active: Optional[ModelHandle] = None

    def list_models(self) -> Dict[str, Dict[str, str]]:
        return self._config.get().get("models", {})

    def get_active(self) -> Optional[ModelHandle]:
        return self._active

    def switch_active(self, model_name: str, backend: str, device: str) -> ModelHandle:
        handle = self.get_model(model_name, backend, device)
        self._active = handle
        return handle

    def get_model(self, model_name: str, backend: str = "auto", device: str = "auto") -> ModelHandle:
        with self._lock:
            resolved_backend = self._resolve_backend(model_name, backend)
            resolved_device = self._resolve_device(resolved_backend, device)
            cache_key = (model_name, resolved_backend, resolved_device)
            if cache_key in self._models:
                return self._models[cache_key]

            model_path = self._get_model_path(model_name, resolved_backend)
            model = self._load_model(model_path, resolved_backend, resolved_device)
            handle = ModelHandle(model_name, resolved_backend, resolved_device, model)
            self._models[cache_key] = handle
            if self._active is None:
                self._active = handle
            return handle

    def _resolve_backend(self, model_name: str, backend: str) -> str:
        if backend != "auto":
            return backend
        candidates = ["tensorrt", "onnx", "pytorch"]
        for candidate in candidates:
            if self._get_model_path(model_name, candidate, silent=True):
                if candidate == "tensorrt" and trt is None:
                    continue
                if candidate == "onnx" and (ort is None and YOLO is None):
                    continue
                if candidate == "pytorch" and YOLO is None:
                    continue
                return candidate
        return "mock"

    def _resolve_device(self, backend: str, device: str) -> str:
        if device != "auto":
            return device
        if backend == "tensorrt":
            return "cuda" if trt is not None else "cpu"
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _get_model_path(self, model_name: str, backend: str, silent: bool = False) -> Optional[str]:
        if backend == "mock":
            return None
        models = self.list_models()
        if model_name not in models:
            if silent:
                return None
            raise ValueError(f"Unknown model: {model_name}")
        path = models[model_name].get(backend)
        if not path:
            if silent:
                return None
            raise ValueError(f"Backend {backend} not available for model {model_name}")
        return path

    def _load_model(self, path: Optional[str], backend: str, device: str) -> Any:
        if backend == "mock" or path is None:
            logger.warning("using_dummy_model", extra={"extra": {"backend": backend}})
            return DummyModel()
        if backend in {"pytorch", "onnx"}:
            if YOLO is None:
                raise RuntimeError("ultralytics is required for pytorch/onnx backends")
            model = YOLO(path)
            return model
        if backend == "tensorrt":
            if trt is None:
                raise RuntimeError("tensorrt is not available")
            return path
        raise ValueError(f"Unsupported backend: {backend}")
