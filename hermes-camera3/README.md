# Gluvok Weighment & ANPR Integration System

An industrial weighing bridge integration service designed to bridge weighing scale indicators, multi-camera capture, ANPR (Automatic Number Plate Recognition) voting, and **Cloud Supabase** database logging.

---

## 🏗️ Project Architecture & Folder Structure

```
hermes-camera2/
├── main.py                    # Application entry point & lifecycle loop
├── config.json                # Local persistent configuration settings
├── requirements.txt           # Python dependencies
├── README.md
├── tests/
│   └── test_scale_uart.py     # Scale UART parser unit tests
└── src/
    ├── config/
    │   ├── config_manager.py  # JSON-backed configuration manager
    │   └── camera_config.py   # IP camera URLs, ANPR server URL, and timers
    ├── scale/
    │   ├── scale_uart.py      # UART serial stream reader & buffer parser
    │   └── scale_stability.py # 10s continuous weight stability state machine
    ├── camera/
    │   ├── camera_manager.py  # HTTP snapshot / RTSP frame grabber
    │   ├── anpr_client.py     # ANPR server client & plate voting algorithm
    │   └── session_manager.py # Weighbridge session lifecycle & image packaging
    └── network/
        ├── supabase_client.py # Supabase URL and Auth token storage
        ├── supabase_auth.py   # Operator JWT login & profile resolver
        └── supabase_post.py   # Weighment payload & base64 image uploader
```

---

## 🌟 Key Features

- **Weight Scale Serial Parsing**: Reads continuous raw serial stream from UART (`/dev/ttyAMA0` or USB-to-Serial at 1200 Baud 8N1).
- **Weight Stabilization Detection**: 10-second continuous weight stability tracking (`STABILITY_TOLERANCE = 2.0 kg`, `STABILITY_DURATION = 10s`).
- **ANPR Multi-Sample Voting**: Captures Camera 1 frames every 2 seconds during active weighing and selects the highest-frequency plate candidate.
- **Concurrent Auxiliary Camera Snapshots**: Captures overview snapshots from auxiliary cameras in parallel upon weight stabilization.
- **Supabase Cloud Integration**: Authenticates with Supabase Auth REST API and posts complete weighment records with base64 images.

---

## 🚀 Running the Application on Raspberry Pi

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure ANPR & Cloud Settings
Set your configuration either via environment variables or in `config.json`:
- `ANPR_SERVER_URL`: URL to Argus FastAPI endpoint (defaults to `http://127.0.0.1:8000/recognize`).
- `ANPR_CAMERA_URL`: Snapshot URL of IP Camera 1.
- `ANPR_SERVER_TIMEOUT`: Timeout in seconds (default `6.0s`).

### 3. Run Tests
```bash
PYTHONPATH=. python3 -m unittest discover tests
```

### 4. Start Application
```bash
python3 main.py
```

