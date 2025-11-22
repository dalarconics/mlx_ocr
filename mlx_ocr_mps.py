#!/usr/bin/env python3
"""
MLX OCR Demo using Hugging Face TrOCR

This script demonstrates optical character recognition (OCR) using
the TrOCR model from Hugging Face, optimized for Apple Silicon with MLX.
"""
import sys
import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

def main():
    if len(sys.argv) != 2:
        print("Usage: python mlx_ocr_mps.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    
    # 1. Use Qwen2-VL (Optimized for OCR and Text Reading)
    # This model is 2B parameters, fits easily in RAM, and runs on Metal
    model_path = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
    
    print(f"Loading model: {model_path}...")
    model, processor = load(model_path)
    config = load_config(model_path)

    # 2. Prepare the prompt
    prompt = "Read all the text in this image word for word."
    
    formatted_prompt = apply_chat_template(
        processor, config, prompt, num_images=1
    )

    # 3. Generate (Inference running on GPU/Metal)
    print(f"Reading {image_path}...")
    output = generate(
        model, 
        processor, 
        formatted_prompt, 
        [image_path], 
        verbose=False
    )
    
    print("\n--- Extracted Text (MLX) ---")
    print(output)
    print("----------------------------")

if __name__ == "__main__":
    main()