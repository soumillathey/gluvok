# Gluvok Weighbridge & ANPR System

Monorepo containing the two core services for the Gluvok industrial weighbridge integration platform.

## 📦 Repository Structure

```
gluvok/
├── argus/           # FastAPI ANPR Microservice (YOLO v11 + Docling OCR)
└── hermes/          # Weighbridge Controller (UART Scale + Multi-Camera + Supabase)
```

## 🚀 Quick Start on Raspberry Pi

### 1. Install `uv` (for Argus)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

### 2. Start Argus ANPR Server
```bash
cd argus
uv sync
uv run python main.py --server --host 0.0.0.0 --port 8000
```

### 3. Start Hermes Weighbridge Controller
```bash
cd hermes
pip install -r requirements.txt
python3 main.py
```

See individual `README.md` files inside each folder for detailed configuration.
