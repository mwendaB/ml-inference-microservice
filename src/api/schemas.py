from typing import List, Optional, Tuple

from pydantic import BaseModel


class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]
    track_id: Optional[int] = None


class InferenceResponse(BaseModel):
    detections: List[Detection]
    inference_time_ms: float
    image_shape: Tuple[int, int]
    model_used: str
    backend: str


class ModelSwitchRequest(BaseModel):
    model_name: str
    backend: str = "auto"
    device: str = "auto"
