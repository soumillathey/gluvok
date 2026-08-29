"""
camera_config.py
Configuration for IP Cameras, ANPR Server, and Session Timers.
Edit camera URLs and settings directly in this file for site-specific deployments.
"""

import os

# ── ANPR Camera (Camera 1) ──────────────────────────────────────────────────
# Camera 1 URL used for continuous 2-second ANPR frame capturing
ANPR_CAMERA_URL = os.getenv(
    "ANPR_CAMERA_URL",
    "http://192.168.1.101/cgi-bin/snapshot.cgi"
)

# ── Auxiliary Overview Cameras (Cameras 2 ... N) ────────────────────────────
# List of URLs for overview cameras captured concurrently when weight stabilizes.
# Set to an empty list [] if only ANPR Camera 1 is used at a site.
AUXILIARY_CAMERA_URLS = [
    "http://192.168.1.102/cgi-bin/snapshot.cgi",  # Camera 2 (e.g. Top / Platform)
    "http://192.168.1.103/cgi-bin/snapshot.cgi",  # Camera 3 (e.g. Front / Side)
    "http://192.168.1.104/cgi-bin/snapshot.cgi",  # Camera 4 (e.g. Rear / Side)
]

# ── ANPR Server Settings ────────────────────────────────────────────────────
# Argus FastAPI ANPR Microservice running locally or over LAN on Raspberry Pi
ANPR_SERVER_URL = os.getenv(
    "ANPR_SERVER_URL",
    "http://127.0.0.1:8000/recognize"
)

# ── Timing & Timeouts ────────────────────────────────────────────────────────
ANPR_CAPTURE_INTERVAL = 2.0        # Seconds between Camera 1 ANPR captures
POST_STABILITY_DURATION = 10.0      # Additional seconds to capture ANPR after stability
CAMERA_TIMEOUT = 3.0               # Seconds allowed for individual camera snapshot HTTP request
ANPR_SERVER_TIMEOUT = 15.0         # Seconds allowed for ANPR HTTP POST request (YOLO+OCR on RPi takes 6-10s)
MAX_PARALLEL_CAMERA_WORKERS = 4    # Maximum concurrent thread pool workers for snapshots

