# MLX Examples: OCR, Sentiment Analysis, and LLMs

A collection of scripts demonstrating various machine learning tasks on Apple Silicon using the MLX framework.

## Description

This project provides simple, easy-to-run Python scripts for a variety of ML tasks. All examples are optimized to run efficiently on Apple Silicon Macs using MLX.

The examples include:
- **Optical Character Recognition (OCR)**: Extract text from images.
- **Sentiment Analysis**: Classify text as positive, negative, or neutral.
- **Text Embeddings**: Generate vector embeddings from text using a BERT model.
- **LLM Inference**: Run various popular large language models.

## Requirements

- Python 3.8+
- An Apple Silicon Mac (M1, M2, M3, etc.)
- Dependencies listed in `requirements.txt`

## Installation

1.  Clone the repository:
    ```bash
    git clone <repository_url>
    cd mlx_ocr
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Demos

### Optical Character Recognition (OCR)

This demo uses a vision-language model to extract text from a given image.

-   **Script:** `mlx_ocr_mps.py`
-   **Model:** `mlx-community/Qwen2-VL-2B-Instruct-4bit`
-   **Usage:**
    ```bash
    python mlx_ocr_mps.py sample.png
    ```

### Sentiment Analysis

This demo classifies a piece of text as 'Positive', 'Negative', or 'Neutral' using a lightweight instruction-tuned model.

-   **Script:** `sentiment.py`
-   **Model:** `mlx-community/Llama-3.2-1B-Instruct-4bit`
-   **Usage:**
    ```bash
    python sentiment.py
    ```

### Text Embeddings with BERT

This script generates a 384-dimensional vector embedding for a sentence using a quantized MiniLM model.

-   **Script:** `mlx-bert.py`
-   **Model:** `mlx-community/all-MiniLM-L6-v2-4bit`
-   **Usage:**
    ```bash
    python mlx-bert.py
    ```

### Large Language Model (LLM) Inference

These scripts demonstrate how to run various popular 4-bit quantized large language models with MLX.

-   **Scripts:**
    -   `mlx-llama31-8B-INS-4B.py`
    -   `mlx-mistral-7B-INS-V03-4B.py`
    -   `mlx-mixtral-8x7-INS-4B.py`
-   **Usage:**
    ```bash
    python <script_name>.py
    ```
    *For example:*
    ```bash
    python mlx-llama31-8B-INS-4B.py
    ```

## License

This project is open source and available under the MIT License.