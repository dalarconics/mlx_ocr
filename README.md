# MLX OCR Demo

A minimal demonstration of optical character recognition (OCR) using Hugging Face's TrOCR model optimized for Apple Silicon with MLX.

## Description

This project provides a simple Python script that uses the TrOCR (Transformer-based OCR) model from Microsoft to extract text from images. The implementation is designed to work efficiently on Apple Silicon Macs using the MLX framework.

## Requirements

- Python 3.7+
- Apple Silicon Mac (for optimal MLX performance)
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/mlx_ocr.git
   cd mlx_ocr
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the OCR demo on an image file:

```bash
python ocr_demo.py sample.png
```

The script will:
- Load the specified image
- Process it with the TrOCR model
- Output the extracted text to the console

## Example Output

```
Processing image: sample.png
Image size: (800, 600)
Loading TrOCR model...
Extracted Text:
==================================================
Hello World! This is a sample text for OCR demonstration.
==================================================
```

## Dependencies

- `transformers`: Hugging Face transformers library for the TrOCR model
- `mlx`: Apple MLX framework for efficient machine learning on Apple Silicon
- `pillow`: Python Imaging Library for image processing
- `torch`: PyTorch (required for transformers compatibility)

## Model

This demo uses the `microsoft/trocr-base-printed` model, which is trained on printed text. For handwritten text recognition, you can modify the script to use `microsoft/trocr-base-handwritten`.

## License

This project is open source and available under the MIT License.