import numpy as np

from src.core.inference_engine import InferenceEngine


def test_infer_with_mock_backend():
    engine = InferenceEngine("yolov8n", backend="mock", device="cpu")
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    result = engine.infer(image)
    assert result["detections"]
    assert result["image_shape"] == (64, 64)


def test_batch_processing():
    engine = InferenceEngine("yolov8n", backend="mock", device="cpu")
    images = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(3)]
    results = engine.infer_batch(images)
    assert len(results) == 3


def test_empty_image_raises():
    engine = InferenceEngine("yolov8n", backend="mock", device="cpu")
    try:
        engine.infer(b"")
    except ValueError:
        assert True
    else:
        assert False
