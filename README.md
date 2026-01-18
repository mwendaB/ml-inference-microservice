# Advanced Multi-Model Object Detection System

## Architecture Overview

```
[RTSP/Files/API] --> [FastAPI] --> [Inference Engine] --> [Model Manager]
                                   |                 --> [PyTorch/ONNX/TensorRT]
                                   |--> [Tracker]
                                   |--> [Metrics -> Prometheus/Redis]
                                   |--> [Dashboard -> Streamlit]
```

## Quick Start (Docker)

```bash
docker-compose up -d
```

API: `http://localhost:8000`
Dashboard: `http://localhost:8501`

## API Documentation

- `POST /api/v1/detect`
  - Upload a single image.
  - Response: `InferenceResponse` with detections and latency.

- `POST /api/v1/detect/batch`
  - Upload multiple images.
  - Response: list of inference results.

- `POST /api/v1/track/video`
  - Upload a video file.
  - Response: path to annotated output.

- `GET /api/v1/models`
  - List available models/backends.

- `POST /api/v1/models/switch`
  - Body: `{ "model_name": "yolov8n", "backend": "onnx", "device": "auto" }`

- `GET /metrics`
  - Prometheus metrics.

- `GET /health`
  - Health check.

- `WebSocket /ws/stream`
  - Send image frames as bytes; receive detections.

## Model Guide

1. Add model files into `models/` (e.g. `yolov8n.pt`, `yolov8n.onnx`).
2. Register in `config/models.yaml` under `models:`.
3. Restart API or hot-reload config with `ConfigLoader`.

## Quantization Tutorial

1. Export ONNX model (use `scripts/export_all_formats.py`).
2. Run quantization:

```bash
python scripts/quantize_models.py --onnx models/yolov8n.onnx --output-dir models
```

3. Review the printed report and JSON output.

## Performance Tuning

- Prefer `tensorrt` backend when available.
- Enable `cuda` device on GPU instances.
- Use batch inference for high throughput.
- Reduce input size for faster CPU inference.

## Deployment Guide

- Configure `.env` values.
- Ensure GPU drivers and CUDA are available.
- Use `docker-compose up -d`.
- Validate `/health` and `/metrics` endpoints.

## Monitoring Setup

- Redis stores recent metrics.
- Prometheus scrapes `/metrics`.
- Streamlit reads from Redis and visualizes latency and usage.

## Troubleshooting

- `ultralytics` missing: install dependencies from `requirements.txt`.
- `tensorrt` unavailable: use `onnx` or `pytorch` backend.
- RTSP errors: verify stream URL and network reachability.

## Benchmarks

Use `scripts/benchmark_comparison.py` to generate JSON metrics across models/backends.

## Roadmap

Future enhancements (documented only):
- Custom model training pipeline
- Active learning loop
- Multi-camera synchronization
- Cloud storage integration (S3)
- Event-driven architecture (Kafka)
- Horizontal scaling (Kubernetes)
- Model versioning (MLflow)
- A/B testing framework
- Mobile deployment (ONNX Runtime Mobile)
- Browser deployment (ONNX.js)
