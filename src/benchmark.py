import argparse
import time
import numpy as np
import onnxruntime as ort
import cv2
import os

def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    return img

def benchmark(model_path, input_path, runs):
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    if not os.path.exists(input_path):
        print(f"Error: Input image not found at {input_path}")
        return
    img = cv2.imread(input_path)
    input_tensor = preprocess(img)
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    latencies = []
    for _ in range(runs):
        start = time.time()
        session.run(None, {input_name: input_tensor})
        latencies.append((time.time() - start) * 1000)
    latencies = np.array(latencies)
    mean = np.mean(latencies)
    median = np.median(latencies)
    p95 = np.percentile(latencies, 95)
    fps = 1000.0 / mean if mean > 0 else 0
    print("\n===== BENCHMARK RESULTS =====")
    print(f"Mean latency:   {mean:.2f} ms")
    print(f"Median latency: {median:.2f} ms")
    print(f"p95 latency:    {p95:.2f} ms")
    print(f"FPS:            {fps:.2f}")
    print(f"Mode:           CPU")
    print("============================\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to ONNX model')
    parser.add_argument('--input', required=True, help='Path to input image')
    parser.add_argument('--runs', type=int, default=50, help='Number of runs')
    args = parser.parse_args()
    benchmark(args.model, args.input, args.runs)

if __name__ == "__main__":
    main()
