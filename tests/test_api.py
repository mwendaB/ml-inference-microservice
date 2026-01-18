from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import io

from src.api.main import app
from src.api import routes
from src.core.inference_engine import InferenceEngine


client = TestClient(app)


def setup_module(module):
    routes._engine = InferenceEngine("yolov8n", backend="mock", device="cpu")


def _create_image_bytes():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    pil_image = Image.fromarray(image)
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG")
    return buffer.getvalue()


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200


def test_detect_endpoint():
    image_bytes = _create_image_bytes()
    response = client.post(
        "/api/v1/detect",
        files={"file": ("test.jpg", image_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert "detections" in payload


def test_detect_invalid_file():
    response = client.post(
        "/api/v1/detect",
        files={"file": ("test.txt", b"invalid", "text/plain")},
    )
    assert response.status_code == 400


def test_models_endpoint():
    response = client.get("/api/v1/models")
    assert response.status_code == 200


def test_switch_model():
    response = client.post(
        "/api/v1/models/switch",
        json={"model_name": "yolov8n", "backend": "mock", "device": "cpu"},
    )
    assert response.status_code == 200
