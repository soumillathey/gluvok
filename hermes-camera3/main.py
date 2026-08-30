"""
main.py — Gluvok Weighment & ANPR Integration
==============================================
Core weighment indicator and ANPR multi-camera capture application.

Module map:
  src/config/    — JSON-backed settings manager & camera configurations
  src/network/   — Supabase login, profile resolver, payload POST
  src/scale/     — PySerial UART stream reader, 10s stability state machine
  src/camera/    — Multi-camera snapshots, ANPR client, session lifecycle
"""

import logging
import signal
import sys
import time

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-5s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ── Import core modules ───────────────────────────────────────────────────────
from src.config.config_manager import config
from src.network.supabase_auth import login_to_supabase
from src.scale.scale_uart import get_uart_reader
from src.web.server import start_web_server, stop_web_server


# ── Graceful shutdown ─────────────────────────────────────────────────────────
def shutdown(signum, frame):
    logger.info("\n[Main] Shutdown signal received. Cleaning up...")
    get_uart_reader().stop()
    stop_web_server()
    sys.exit(0)


signal.signal(signal.SIGINT,  shutdown)
signal.signal(signal.SIGTERM, shutdown)

# ─────────────────────────────────────────────────────────────────────────────
#  SETUP
# ─────────────────────────────────────────────────────────────────────────────
def setup():
    logger.info("")
    logger.info("==============================================")
    logger.info("Gluvok Weighment & ANPR System Starting...")
    logger.info("==============================================")

    # Start UART scale reader thread
    get_uart_reader().start()

    # Start fallback diagnostics and Wi-Fi configuration web server
    start_web_server(port=8080)

    # Log active settings from config.json
    logger.info(
        f"[Config] Email: '{config.supabase_email}' | "
        f"Center ID: {config.supabase_center_id} | "
        f"Threshold: {config.supabase_weight_threshold:.1f} kg"
    )

    # Authenticate with Supabase backend
    if config.supabase_email and config.supabase_password:
        login_to_supabase()
    else:
        logger.info("[Auth] Supabase credentials not set in config.json.")


# ─────────────────────────────────────────────────────────────────────────────
#  LOOP
# ─────────────────────────────────────────────────────────────────────────────
def loop():
    time.sleep(1)

# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    setup()
    while True:
        try:
            loop()
        except KeyboardInterrupt:
            break
