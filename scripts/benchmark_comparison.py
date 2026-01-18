import argparse
import json
import time
from pathlib import Path
from typing import Dict

from src.core.inference_engine import InferenceEngine
from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import cv2
except Exception:
    cv2 = None


def benchmark_video(engine: InferenceEngine, video_path: str, runs: int) -> Dict[str, float]:
    if cv2 is None:
        raise RuntimeError("opencv is required for benchmarking")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Unable to open video")

    latencies = []
    frames = 0
    start_total = time.perf_counter()
    while frames < runs:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        start = time.perf_counter()
        _ = engine.infer(frame)
        latencies.append((time.perf_counter() - start) * 1000)
        frames += 1
    total_time = time.perf_counter() - start_total
    cap.release()

    fps = frames / total_time if total_time > 0 else 0
    latencies_sorted = sorted(latencies)
    return {
        "fps": fps,
        "latency_p50": latencies_sorted[int(0.5 * len(latencies_sorted))],
        "latency_p95": latencies_sorted[int(0.95 * len(latencies_sorted))],
        "latency_p99": latencies_sorted[int(0.99 * len(latencies_sorted))],
    }


def _write_markdown(report: Dict[str, Dict[str, Dict[str, float]]], path: str) -> None:
    lines = [
        "# Benchmark Report",
        "",
        "| Model | Backend | FPS | p50 (ms) | p95 (ms) | p99 (ms) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for model, backends in report.items():
        for backend, metrics in backends.items():
            lines.append(
                f"| {model} | {backend} | {metrics['fps']:.2f} | {metrics['latency_p50']:.2f} | {metrics['latency_p95']:.2f} | {metrics['latency_p99']:.2f} |"
            )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def _write_charts(report: Dict[str, Dict[str, Dict[str, float]]], prefix: Path) -> None:
    try:
        import plotly.express as px
    except Exception:
        return

    rows = []
    for model, backends in report.items():
        for backend, metrics in backends.items():
            rows.append({"model": model, "backend": backend, "fps": metrics["fps"], "p95": metrics["latency_p95"]})

    if rows:
        fig = px.bar(rows, x="model", y="fps", color="backend", barmode="group", title="FPS Comparison")
        fig.write_html(str(prefix) + "_fps.html")
        fig = px.box(rows, x="backend", y="p95", color="model", title="Latency p95")
        fig.write_html(str(prefix) + "_latency.html")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark models across backends")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--backends", nargs="+", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--output", default="benchmark_report.json")
    parser.add_argument("--markdown", default="benchmark_report.md")
    args = parser.parse_args()

    report: Dict[str, Dict[str, Dict[str, float]]] = {}
    for model in args.models:
        report[model] = {}
        for backend in args.backends:
            engine = InferenceEngine(model, backend, device="auto")
            metrics = benchmark_video(engine, args.input, args.runs)
            report[model][backend] = metrics
            logger.info("benchmark_complete", extra={"extra": {"model": model, "backend": backend}})

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    _write_markdown(report, args.markdown)
    _write_charts(report, Path(args.output).with_suffix(""))

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
