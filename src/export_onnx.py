import sys
from ultralytics import YOLO
import os

ONNX_PATH = 'models/yolov8n.onnx'

def export_onnx():
    try:
        model = YOLO('yolov8n.pt')
    except Exception as e:
        print(f"Error: Failed to load YOLOv8n model. {e}")
        sys.exit(1)
    try:
        model.export(format='onnx', dynamic=False, simplify=True, imgsz=640, half=False, device='cpu',
                     export_path=ONNX_PATH)
    except Exception as e:
        print(f"Error: Failed to export ONNX. {e}")
        sys.exit(1)
    if os.path.exists(ONNX_PATH):
        print(f"ONNX model exported: {ONNX_PATH}")
    else:
        print("Error: ONNX export failed.")
        sys.exit(1)

if __name__ == "__main__":
    export_onnx()
