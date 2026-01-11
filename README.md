# Real-Time Object Detection + ONNX Export + Benchmark Pipeline

## Project Purpose
A modular Python pipeline for static and real-time object detection using YOLOv8n, ONNX export, ONNX Runtime inference, and benchmarking on CPU. No training or GPU code included.

## Setup Instructions
1. Clone/download this repository.
2. Ensure Python 3.10+ is installed.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Directory Structure
```
assets/
  test.jpg           # Place a test image here
models/
  yolov8n.onnx       # Will be created after ONNX export
src/
  inference.py       # Static image inference
  webcam.py          # Real-time webcam inference
  export_onnx.py     # Export YOLOv8n to ONNX
  onnx_inference.py  # ONNX Runtime inference
  benchmark.py       # CLI benchmarking
README.md
requirements.txt
SPEC.md
```

## How to Run Each Phase

### 1. Static Image Inference
```bash
python src/inference.py
```

### 2. Webcam Real-Time Inference
```bash
python src/webcam.py
```

### 3. Export Model to ONNX
```bash
python src/export_onnx.py
```

### 4. ONNX Runtime Inference
```bash
python src/onnx_inference.py
```

### 5. CLI Benchmark Script
```bash
python src/benchmark.py --model models/yolov8n.onnx --input assets/test.jpg --runs 50
```

## Benchmark Usage
- Prints mean, median, p95 latency, FPS, and mode (CPU) in a clean block.

## Optional Future Improvements
- TensorRT optimization
- Quantization (FP16/INT8)
- Cloud GPU benchmarking
- FastAPI inference service
- RTSP streaming
- Docker deployment

## Notes
- All code runs on CPU only.
- Each script is independent.
- No CUDA/MPS/TensorRT code included.
- No global variables or duplicate inference logic.
- Place your test image as `assets/test.jpg` before running.
