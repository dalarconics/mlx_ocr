#!/usr/bin/env python3
"""
MLX OCR Demo using Hugging Face TrOCR

This script demonstrates optical character recognition (OCR) using
the TrOCR model from Hugging Face, optimized for Apple Silicon with MLX.
"""

import sys
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

def main():
    # Check if image path is provided
    if len(sys.argv) != 2:
        print("Usage: python ocr_demo.py <image_path>")
        print("Example: python ocr_demo.py sample.png")
        sys.exit(1)

    image_path = sys.argv[1]

    try:
        # Load the image
        image = Image.open(image_path).convert('RGB')
        print(f"Processing image: {image_path}")
        print(f"Image size: {image.size}")

        # Load the TrOCR processor and model
        print("Loading TrOCR model...")
        processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
        model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')

        # Process the image
        pixel_values = processor(image, return_tensors="pt").pixel_values

        # Generate text
        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        print("\nExtracted Text:")
        print("=" * 50)
        print(generated_text)
        print("=" * 50)

    except FileNotFoundError:
        print(f"Error: Image file '{image_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()