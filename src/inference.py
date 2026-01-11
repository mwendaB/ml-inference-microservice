import sys
import time
import cv2
from ultralytics import YOLO

ASSET_PATH = 'assets/test.jpg'


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        print(f"Error: Image not found at {path}")
        sys.exit(1)
    return img


def run_inference():
    try:
        model = YOLO('yolov8n.pt')
    except Exception as e:
        print(f"Error: Failed to load YOLOv8n model. {e}")
        sys.exit(1)

    img = load_image(ASSET_PATH)
    start = time.time()
    results = model(img, device='cpu')
    latency = (time.time() - start) * 1000
    annotated = results[0].plot()
    cv2.imshow('YOLOv8n Inference', annotated)
    print(f"Latency: {latency:.2f} ms")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_inference()
