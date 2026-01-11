import sys
import time
import cv2
import numpy as np
import onnxruntime as ort
import os

ASSET_PATH = 'assets/test.jpg'
ONNX_PATH = 'models/yolov8n.onnx'


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        print(f"Error: Image not found at {path}")
        sys.exit(1)
    return img


def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    return img


def run_onnx_inference():
    if not os.path.exists(ONNX_PATH):
        print(f"Error: ONNX model not found at {ONNX_PATH}")
        sys.exit(1)
    img = load_image(ASSET_PATH)
    input_tensor = preprocess(img)
    session = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    start = time.time()
    outputs = session.run(None, {input_name: input_tensor})
    latency = (time.time() - start) * 1000
    # NOTE: Drawing boxes is not implemented here due to ONNX output complexity
    # For demo, just show the image
    cv2.imshow('ONNX Inference (no boxes)', img)
    print(f"Latency: {latency:.2f} ms")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_onnx_inference()
