import argparse
import json
from pathlib import Path

from src.core.quantizer import quantize_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch quantize ONNX models")
    parser.add_argument("--onnx", required=True, help="Input ONNX model path")
    parser.add_argument("--output-dir", default="models", help="Output directory")
    parser.add_argument("--report", default="quant_report.json", help="Report output path")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fp16_path = output_dir / (Path(args.onnx).stem + "_fp16.onnx")
    int8_path = output_dir / (Path(args.onnx).stem + "_int8.onnx")
    trt_path = output_dir / (Path(args.onnx).stem + ".engine")

    report = quantize_model(args.onnx, str(fp16_path), str(int8_path), str(trt_path))
    print(report.format_report())

    with open(args.report, "w", encoding="utf-8") as handle:
        json.dump(report.__dict__, handle, indent=2)


if __name__ == "__main__":
    main()
