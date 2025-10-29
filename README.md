# MLX OCR Demo

A minimal demonstration of optical character recognition (OCR) using a Hugging Face TrOCR checkpoint converted to the MLX runtime on Apple Silicon.

## Description

This project provides a simple Python script that uses the TrOCR (Transformer-based OCR) model from Microsoft to extract text from images. The original PyTorch checkpoint is converted into MLX weights so inference can run natively on Apple Silicon hardware.

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

3. (One time) install the extra tools required for weight conversion:
   ```bash
   pip install torch safetensors
   ```

4. Convert the Hugging Face checkpoint to MLX weights:
   ```bash
   python convert_to_mlx.py \
     --model microsoft/trocr-base-printed \
     --output weights/trocr-base-printed.npz \
     --config-output configs/trocr-base-printed.json
   ```

## Usage

Run the OCR demo on an image file:

```bash
python ocr_demo.py sample.png --weights weights/trocr-base-printed.npz
```

The script will:
- Load the specified image
- Process it with the MLX TrOCR model
- Output the extracted text to the console

## Example Output

```
Processing image: sample.png
Image size: (800, 600)
Extracted Text:
==================================================
Hello World! This is a sample text for OCR demonstration.
==================================================
```

## Dependencies

- `transformers`: Hugging Face toolkit for preprocessing/tokenization
- `mlx`: Apple MLX framework for efficient ML inference on Apple Silicon
- `pillow`: Python Imaging Library for image processing
- `numpy`: NumPy array utilities used during preprocessing
- Optional (conversion only): `torch`, `safetensors`

## Model

This demo uses the `microsoft/trocr-base-printed` model, which is trained on printed text. For handwritten text recognition, you can modify the script to use `microsoft/trocr-base-handwritten`.

## License

This project is open source and available under the MIT License.
