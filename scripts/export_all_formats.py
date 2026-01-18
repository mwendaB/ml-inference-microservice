import argparse
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLO model to ONNX and TensorRT")
    parser.add_argument("--model", required=True, help="Path to .pt model")
    parser.add_argument("--output-dir", default="models", help="Output directory")
    args = parser.parse_args()

    if YOLO is None:
        raise RuntimeError("ultralytics is required for export")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    model.export(format="onnx", simplify=True, imgsz=640, opset=12)
    model.export(format="engine", half=True, imgsz=640)


if __name__ == "__main__":
    main()
