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


def draw_fps(frame, fps):
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    return frame


def webcam_inference():
    try:
        model = YOLO('yolov8n.pt')
    except Exception as e:
        print(f"Error: Failed to load YOLOv8n model. {e}")
        sys.exit(1)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Warning: Webcam not found. Looping test image.")
        img = load_image(ASSET_PATH)
        while True:
            start = time.time()
            results = model(img, device='cpu')
            annotated = results[0].plot()
            fps = 1.0 / (time.time() - start)
            annotated = draw_fps(annotated, fps)
            cv2.imshow('YOLOv8n Webcam (Fallback)', annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        start = time.time()
        results = model(frame, device='cpu')
        annotated = results[0].plot()
        fps = 1.0 / (time.time() - start)
        annotated = draw_fps(annotated, fps)
        cv2.imshow('YOLOv8n Webcam', annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    webcam_inference()
