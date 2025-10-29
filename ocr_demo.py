#!/usr/bin/env python3
"""
MLX OCR Demo using a converted TrOCR checkpoint.

This script loads MLX weights exported from the Hugging Face TrOCR model and
performs greedy decoding on an input image.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx
from PIL import Image
from transformers import TrOCRProcessor

from mlx_ocr import MLXTrOCRModel, load_config_dict

DEFAULT_CONFIG = Path("configs/trocr-base-printed.json")
DEFAULT_WEIGHTS = Path("weights/trocr-base-printed.npz")


def load_image(image_path: Path) -> Image.Image:
    image = Image.open(image_path)
    return image.convert("RGB")


def prepare_inputs(processor: TrOCRProcessor, image: Image.Image) -> mx.array:
    pixel_values = processor(images=image, return_tensors="np").pixel_values
    pixel_values = pixel_values.transpose(0, 2, 3, 1)
    return mx.array(pixel_values, dtype=mx.float32)


def decode_sequences(sequences, processor: TrOCRProcessor):
    # Drop the initial decoder start token
    trimmed = [seq[1:] for seq in sequences]
    return processor.batch_decode(trimmed, skip_special_tokens=True)


def run_demo(args):
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file '{image_path}' not found.")

    if not args.weights.exists():
        raise FileNotFoundError(
            f"MLX weight file '{args.weights}' not found. "
            "Run convert_to_mlx.py first."
        )

    processor = TrOCRProcessor.from_pretrained(args.model_name)
    config = load_config_dict(args.config)
    model = MLXTrOCRModel(config)
    model.load_weights(str(args.weights))
    model.eval()
    mx.eval(model.parameters())

    image = load_image(image_path)
    print(f"Processing image: {image_path}")
    print(f"Image size: {image.size}")

    pixel_values = prepare_inputs(processor, image)
    generated = model.generate(pixel_values, max_length=args.max_length)
    generated = mx.eval(generated)
    sequences = generated.tolist()
    texts = decode_sequences(sequences, processor)

    print("\nExtracted Text:")
    print("=" * 50)
    print(texts[0])
    print("=" * 50)


def parse_args():
    parser = argparse.ArgumentParser(description="Run OCR with MLX TrOCR.")
    parser.add_argument("image", help="Path to the input image.")
    parser.add_argument(
        "--model-name",
        default="microsoft/trocr-base-printed",
        help="Processor identifier for tokenization and feature extraction.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to the MLX configuration JSON.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="Path to the MLX weights (.npz).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Optional override for maximum decoding length.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        run_demo(args)
    except Exception as exc:
        raise SystemExit(f"Error: {exc}")


if __name__ == "__main__":
    main()
