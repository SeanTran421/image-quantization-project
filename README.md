# MNIST Image Classifier Quantization with TensorFlow Lite

This project demonstrates model quantization of a simple MNIST image classifier using TensorFlow Lite. It explores dynamic range, float16, and full integer (int8) quantization techniques and compares their impact on accuracy, model size, and inference speed.

## Project Structure
- **image-quantization-project/**
  - `main.py` – Train and convert the MNIST model
  - `evaluate.py` – Evaluate accuracy, latency, and model size
  - **models/**
    - `original_model.h5`
    - `quantized_model.tflite`
    - `int8_quantized_model.tflite`
    - `float16_quantized_model.tflite`
  - `README.md` – This file
  - `requirements.txt` – Python dependencies
  - `.gitignore` – Exclude venv, __pycache__, etc.

## Results

| Model Type                | Model Size (MB) | Accuracy (%) | Inference Time (s) |
|---------------------------|----------------|--------------|------------------|
| Original (FP32)           | 1.19           | 97.71        | 0.00             |
| Dynamic Range (TFLite)    | 0.10           | 97.71        | 0.08             |
| Full Integer (INT8)       | 0.10           | 97.69        | 0.12             |
| Float16 (TFLite)          | 0.20           | 97.71        | 0.14             |

## Observations

- Quantization reduces model size drastically with minimal accuracy loss (<0.1%).
- Dynamic range is fastest and smallest; INT8 slightly slower; float16 is a tradeoff between speed and numerical precision.

## How to Run

1. **Create and activate a virtual environment:**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```
3. **Train and convert models:**

```bash
python main.py
```
4. **Evaluate models:**

```bash
python evaluate.py
```
**Key Learnings:**

Training a Keras MNIST model and converting it to TFLite.

Applying dynamic range, float16, and INT8 quantization.

Measuring model size, accuracy, and inference latency.

Gaining hands-on experience in edge-device optimization and Python evaluation pipelines.