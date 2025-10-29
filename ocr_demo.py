#!/usr/bin/env python3
"""
MLX OCR Demo using Hugging Face TrOCR

This script demonstrates optical character recognition (OCR) using
the TrOCR model from Hugging Face, optimized for Apple Silicon with MLX.
"""

#!/usr/bin/env python3
"""
MLX OCR Demo using Hugging Face TrOCR

This script demonstrates optical character recognition (OCR) using
the TrOCR model from Hugging Face, optimized for Apple Silicon with MLX.
"""

import sys
import os
from PIL import Image

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

        # Set environment variables to avoid threading issues
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        print("Loading TrOCR model...")
        # Try importing after setting environment variables
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            import torch

            processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
            model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')

            # Force CPU usage to avoid GPU/threading issues
            device = torch.device('cpu')
            model.to(device)

            # Process the image
            pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

            # Generate text with explicit device
            with torch.no_grad():
                generated_ids = model.generate(pixel_values, max_length=128, num_beams=1)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            print("\nExtracted Text:")
            print("=" * 50)
            print(generated_text)
            print("=" * 50)

        except ImportError as ie:
            print(f"Import error: {ie}")
            print("This may indicate an incompatible environment for MLX/PyTorch integration.")
            sys.exit(1)

    except FileNotFoundError:
        print(f"Error: Image file '{image_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()