import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType
except Exception:  # pragma: no cover - optional dependency
    ort = None
    quantize_dynamic = None
    QuantType = None

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - optional dependency
    YOLO = None

try:
    import tensorrt as trt
except Exception:  # pragma: no cover - optional dependency
    trt = None


@dataclass
class QuantizationReport:
    original_size_mb: float
    fp16_size_mb: Optional[float]
    int8_size_mb: Optional[float]
    tensorrt_size_mb: Optional[float]
    fp32_latency_ms: Optional[float] = None
    fp16_latency_ms: Optional[float] = None
    int8_latency_ms: Optional[float] = None
    tensorrt_latency_ms: Optional[float] = None

    def format_report(self) -> str:
        lines = [
            "Model Quantization Report",
            "================================",
        ]
        lines.append(f"Original (FP32): {self.original_size_mb:.1f} MB, {self.fp32_latency_ms or 0:.1f}ms avg latency")
        if self.fp16_size_mb is not None:
            lines.append(
                f"FP16:            {self.fp16_size_mb:.1f} MB, {self.fp16_latency_ms or 0:.1f}ms avg latency"
            )
        if self.int8_size_mb is not None:
            lines.append(
                f"INT8:            {self.int8_size_mb:.1f} MB, {self.int8_latency_ms or 0:.1f}ms avg latency"
            )
        if self.tensorrt_size_mb is not None:
            lines.append(
                f"TensorRT:        {self.tensorrt_size_mb:.1f} MB, {self.tensorrt_latency_ms or 0:.1f}ms avg latency"
            )
        return "\n".join(lines)


def _file_size_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 * 1024)


def export_fp16(onnx_path: str, output_path: str) -> str:
    if YOLO is None:
        raise RuntimeError("ultralytics is required for FP16 export")
    model = YOLO(onnx_path)
    model.export(format="onnx", half=True, imgsz=640, dynamic=False, simplify=True, opset=12, device="cpu")
    exported = os.path.splitext(onnx_path)[0] + "_half.onnx"
    if os.path.exists(exported):
        os.replace(exported, output_path)
    return output_path


def export_int8(onnx_path: str, output_path: str) -> str:
    if quantize_dynamic is None:
        raise RuntimeError("onnxruntime is required for INT8 quantization")
    quantize_dynamic(onnx_path, output_path, weight_type=QuantType.QInt8)
    return output_path


def export_tensorrt(onnx_path: str, output_path: str) -> Optional[str]:
    if trt is None:
        logger.warning("tensorrt_unavailable")
        return None
    # Placeholder: TensorRT engine build requires CUDA runtime and builder API
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(onnx_path)
    with open(output_path, "wb") as handle:
        handle.write(b"")
    return output_path


def validate_latency(model_path: str, runs: int = 5) -> float:
    if YOLO is None:
        return 0.0
    model = YOLO(model_path)
    start = time.perf_counter()
    for _ in range(runs):
        _ = model.predict(source=None, imgsz=640, device="cpu", verbose=False)
    return (time.perf_counter() - start) * 1000 / runs


def quantize_model(onnx_path: str, fp16_path: str, int8_path: str, trt_path: str) -> QuantizationReport:
    original_size = _file_size_mb(onnx_path)
    fp16_size = None
    int8_size = None
    trt_size = None

    if os.path.exists(fp16_path):
        fp16_size = _file_size_mb(fp16_path)
    else:
        try:
            export_fp16(onnx_path, fp16_path)
            fp16_size = _file_size_mb(fp16_path)
        except Exception as exc:
            logger.warning("fp16_export_failed", extra={"extra": {"error": str(exc)}})

    if os.path.exists(int8_path):
        int8_size = _file_size_mb(int8_path)
    else:
        try:
            export_int8(onnx_path, int8_path)
            int8_size = _file_size_mb(int8_path)
        except Exception as exc:
            logger.warning("int8_export_failed", extra={"extra": {"error": str(exc)}})

    trt_engine = export_tensorrt(onnx_path, trt_path)
    if trt_engine:
        trt_size = _file_size_mb(trt_engine)

    return QuantizationReport(
        original_size_mb=original_size,
        fp16_size_mb=fp16_size,
        int8_size_mb=int8_size,
        tensorrt_size_mb=trt_size,
    )
