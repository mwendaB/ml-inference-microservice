import os
import time
from collections import defaultdict, deque
from typing import Dict, List

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from src.api.schemas import InferenceResponse, ModelSwitchRequest
from src.core.inference_engine import InferenceEngine
from src.core.model_manager import ModelManager
from src.core.tracker import ByteTrack
from src.monitoring.metrics import MetricsRecorder
from src.streaming.video_processor import VideoProcessor
from src.utils.config_loader import ConfigLoader
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1")

_config = ConfigLoader("config/api.yaml").get().get("api", {})
_metrics = MetricsRecorder()
_manager = ModelManager()
_engine = InferenceEngine(
    model_name=_config.get("default_model", "yolov8n"),
    backend=_config.get("default_backend", "onnx"),
    device=_config.get("default_device", "auto"),
)
_tracker = ByteTrack()

_rate_state: Dict[str, deque] = defaultdict(lambda: deque(maxlen=200))


def _rate_limit(client_id: str = "anonymous") -> None:
    limit = _config.get("rate_limit_per_minute", 120)
    if not limit:
        return
    now = time.time()
    window = _rate_state[client_id]
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= limit:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    window.append(now)


def _api_key_guard(api_key: str = None) -> None:
    expected = _config.get("api_keys", [])
    if not expected:
        return
    if api_key is None or api_key not in expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


@router.post("/detect", response_model=InferenceResponse)
async def detect_image(file: UploadFile = File(...)) -> InferenceResponse:
    _rate_limit()
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image file")
    image_bytes = await file.read()
    result = await _engine.infer_async(image_bytes)
    _metrics.record_inference(result["inference_time_ms"], result["model_used"], result["backend"])
    return result


@router.post("/detect/batch")
async def detect_batch(files: List[UploadFile] = File(...)) -> JSONResponse:
    _rate_limit()
    results = []
    for file in files:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Invalid image file")
        image_bytes = await file.read()
        result = await _engine.infer_async(image_bytes)
        _metrics.record_inference(result["inference_time_ms"], result["model_used"], result["backend"])
        results.append(result)
    return JSONResponse(content=results)


@router.post("/track/video")
async def track_video(file: UploadFile = File(...)) -> JSONResponse:
    _rate_limit()
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Invalid video file")
    contents = await file.read()
    temp_path = "./assets/temp_video.mp4"
    os.makedirs("./assets", exist_ok=True)
    with open(temp_path, "wb") as handle:
        handle.write(contents)
    processor = VideoProcessor(_engine.model_name, _engine.backend)
    output_path = processor.process(temp_path)
    return JSONResponse(content={"output_path": output_path})


@router.get("/models")
async def list_models() -> JSONResponse:
    return JSONResponse(content=_manager.list_models())


@router.post("/models/switch")
async def switch_model(payload: ModelSwitchRequest) -> JSONResponse:
    handle = _manager.switch_active(payload.model_name, payload.backend, payload.device)
    _engine.model_name = handle.model_name
    _engine.backend = handle.backend
    _engine.device = handle.device
    return JSONResponse(content={"active_model": handle.model_name, "backend": handle.backend})


__all__ = ["router", "_engine", "_manager"]
