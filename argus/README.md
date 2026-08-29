# Argus ANPR Engine

An Enterprise Automatic Number Plate Recognition (ANPR) Python engine built with **YOLO v11**, **Docling OCR (RapidOCR ONNX Runtime)**, and **Strategy & Factory Design Patterns**.

It features an intelligent **YOLO v11 Pre-screening Pipeline** to verify 4-wheeler vehicle presence (`car`, `bus`, `truck`) before routing to downstream OCR / Vision AI models (**Docling Strategy**, **NVIDIA Llama-3.2-11b-Vision**, or **Plate Recognizer**).

---

## ⚙️ Environment Variables (`.env`)

Set the following environment variables in your local `.env` file:

```env
PLATE_RECOGNIZER_TOKEN=your_plate_recognizer_api_token
LLAMA_API_KEY=your_llama_api_key
NEMOTRON_API_KEY=your_nemotron_api_key
NVIDIA_INVOKE_URL=https://integrate.api.nvidia.com/v1/chat/completions
DEFAULT_PROVIDER=docling
YOLO_MODEL_NAME=yolo11n.pt
YOLO_CONFIG_DIR=/tmp/Ultralytics
HUMAN_CONF_THRESH=0.30
VEHICLE_CONF_THRESH=0.35
```

---

## 🛠️ Installation & Usage

### 1. Installation
```bash
uv sync
```

### 2. Start FastAPI REST API Server
```bash
uv run uvicorn app.server:app --reload --host 0.0.0.0 --port 8000
```
- **Interactive Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc UI**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

#### Example API Requests:

- **Health Check**:
  ```bash
  curl http://localhost:8000/health
  ```

- **Recognize License Plate from Image**:
  ```bash
  curl -X POST "http://localhost:8000/recognize" \
    -F "file=@path/to/vehicle.jpg"
  ```

### 3. Run License Plate Recognition via CLI
```bash
uv run python -m app.main path/to/image.jpg
```

### 4. Use as a Python Library
```python
from app.services.pipeline import recognize_plate_image

# Process image file or raw bytes
response = recognize_plate_image("path/to/image.jpg")

if response.success:
    for plate in response.results:
        print(f"Plate: {plate.plate}, State: {plate.state}")
else:
    print(f"Failed: {response.status_message}")
```

---

## 🧪 Direct Strategy Testing

### Run Direct Strategy Benchmark
```bash
uv run python test_direct.py docling
```

