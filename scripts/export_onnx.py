import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


ASSET_SUFFIXES = {".json", ".txt", ".model"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a Transformers sequence-classification model to ONNX for Transformers.js.",
    )
    parser.add_argument("--model_dir", default="outputs/transformer_deploy")
    parser.add_argument("--out_dir", default="web/models/emoji-model")
    parser.add_argument(
        "--opset",
        type=int,
        default=None,
        help="Optional ONNX opset version. If omitted, transformers.onnx defaults are used.",
    )
    parser.add_argument(
        "--quantize",
        default="q8",
        choices=["q8", "none"],
        help="Quantize to int8 for browser use. Use 'none' to skip quantization.",
    )
    return parser.parse_args()


def run_export(model_dir: Path, out_dir: Path, opset: Optional[int]) -> None:
    if export_with_optimum(model_dir, out_dir):
        if opset is not None:
            print("Note: --opset is ignored when exporting with Optimum.")
        return

    try:
        import transformers.onnx  # noqa: F401
    except Exception as exc:
        if export_with_torch(model_dir, out_dir, opset):
            return
        raise RuntimeError(
            "transformers.onnx is unavailable and Optimum export is not installed. "
            "Install Optimum ONNX Runtime support (pip install optimum-onnxruntime) "
            "or Transformers ONNX extras (pip install \"transformers[onnx]\") "
            "then retry."
        ) from exc

    cmd = [
        sys.executable,
        "-m",
        "transformers.onnx",
        "--model",
        str(model_dir),
        "--feature",
        "sequence-classification",
    ]
    if opset is not None:
        cmd += ["--opset", str(opset)]
    cmd.append(str(out_dir))
    subprocess.run(cmd, check=True)
    relocate_onnx_files(out_dir)


def export_with_optimum(model_dir: Path, out_dir: Path) -> bool:
    try:
        from optimum.onnxruntime import ORTModelForSequenceClassification
        from transformers import AutoTokenizer
    except Exception:
        return False

    out_dir.mkdir(parents=True, exist_ok=True)
    print("Using Optimum to export ONNX.")
    ort_model = ORTModelForSequenceClassification.from_pretrained(
        model_dir,
        export=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    ort_model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    relocate_onnx_files(out_dir)
    return True


def export_with_torch(model_dir: Path, out_dir: Path, opset: Optional[int]) -> bool:
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception:
        return False

    out_dir.mkdir(parents=True, exist_ok=True)
    print("Using torch.onnx.export fallback.")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            attn_implementation="eager",
        )
    except TypeError:
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    sample_texts = [
        "export",
        "export with extra tokens to force padding in the attention mask",
    ]
    encoded = tokenizer(
        sample_texts,
        padding="max_length",
        max_length=16,
        truncation=True,
        return_tensors="pt",
    )
    input_names = [name for name in ("input_ids", "attention_mask") if name in encoded]
    if not input_names:
        raise RuntimeError("Tokenizer did not return any model inputs.")

    inputs = {name: encoded[name] for name in input_names}
    dynamic_axes = {name: {0: "batch", 1: "sequence"} for name in input_names}
    dynamic_axes["logits"] = {0: "batch"}

    onnx_dir = out_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    export_kwargs = {
        "export_params": True,
        "do_constant_folding": True,
        "input_names": input_names,
        "output_names": ["logits"],
        "dynamic_axes": dynamic_axes,
        "external_data": False,
    }
    if opset is not None:
        export_kwargs["opset_version"] = opset

    out_path = onnx_dir / "model.onnx"
    torch.onnx.export(model, args=(), kwargs=inputs, f=out_path, **export_kwargs)
    return True


def copy_assets(model_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in model_dir.iterdir():
        if path.is_file() and path.suffix in ASSET_SUFFIXES:
            shutil.copy2(path, out_dir / path.name)


def quantize_q8(out_dir: Path) -> bool:
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except Exception:
        return False

    model_path = out_dir / "onnx" / "model.onnx"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing ONNX file: {model_path}")
    quantized_path = out_dir / "onnx" / "model_quantized.onnx"
    quantize_dynamic(str(model_path), str(quantized_path), weight_type=QuantType.QInt8)
    return True


def relocate_onnx_files(out_dir: Path) -> None:
    onnx_dir = out_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    mappings = {
        "model.onnx": "model.onnx",
        "model.onnx_data": "model.onnx_data",
        "model_q8.onnx": "model_quantized.onnx",
        "model_q8.onnx_data": "model_quantized.onnx_data",
        "model_quantized.onnx": "model_quantized.onnx",
        "model_quantized.onnx_data": "model_quantized.onnx_data",
    }
    for src_name, dest_name in mappings.items():
        src = out_dir / src_name
        if not src.exists():
            continue
        dest = onnx_dir / dest_name
        if dest.exists():
            dest.unlink()
        shutil.move(str(src), str(dest))


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"model_dir not found: {model_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exporting ONNX from {model_dir} -> {out_dir}")
    run_export(model_dir, out_dir, args.opset)
    copy_assets(model_dir, out_dir)

    if args.quantize == "q8":
        print("Quantizing to int8 (model_q8.onnx)")
        if not quantize_q8(out_dir):
            print(
                "onnxruntime not available; skipping quantization. "
                "Install onnxruntime to generate model_q8.onnx.",
            )

    print("Done.")


if __name__ == "__main__":
    main()
