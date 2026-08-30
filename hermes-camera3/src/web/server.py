import json
import logging
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import requests

from ..config.camera_config import (
    ANPR_CAMERA_URL,
    ANPR_SERVER_URL,
    AUXILIARY_CAMERA_URLS,
)
from ..config.config_manager import config
from ..network.supabase_client import auth_state
from ..scale.scale_stability import get_scale_state
from ..scale.scale_uart import SCALE_BAUD_RATE, SCALE_SERIAL_PORT

logger = logging.getLogger(__name__)

TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")
INDEX_HTML_PATH = os.path.join(TEMPLATES_DIR, "index.html")

# In-memory recent system events buffer for telemetry feed
_system_events: list[dict[str, str]] = []
_events_lock = threading.Lock()


def record_system_event(source: str, message: str):
    """Appends an event to the circular telemetry log (max 20 entries)."""
    with _events_lock:
        if len(_system_events) >= 20:
            _system_events.pop(0)
        _system_events.append({
            "time": time.strftime("%H:%M:%S"),
            "source": source,
            "message": message,
        })


class FallbackHTTPRequestHandler(BaseHTTPRequestHandler):
    """Request handler serving the fallback Tailwind CSS console and REST APIs."""

    def log_message(self, format: str, *args: Any):
        # Silence default standard HTTP access logs to keep terminal clean
        pass

    def _send_json_response(self, data: dict[str, Any], status_code: int = 200):
        response_bytes = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(response_bytes)

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            self._serve_index_html()
        elif self.path == "/api/status":
            self._handle_get_status()
        else:
            self.send_error(404, "Endpoint not found")

    def do_POST(self):
        if self.path == "/api/wifi":
            self._handle_post_wifi()
        elif self.path == "/api/wifi/clear":
            self._handle_post_wifi_clear()
        else:
            self.send_error(404, "Endpoint not found")

    def _serve_index_html(self):
        if not os.path.exists(INDEX_HTML_PATH):
            self.send_error(500, "Dashboard index.html template missing")
            return

        try:
            with open(INDEX_HTML_PATH, "rb") as f:
                content = f.read()

            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except OSError as e:
            self.send_error(500, f"Error reading index.html: {e}")

    def _handle_get_status(self):
        # Check Argus health on port 8000
        target_argus_url = config.anpr_server_url or ANPR_SERVER_URL
        argus_health_url = target_argus_url.replace("/recognize", "/health")
        argus_online = False
        try:
            r = requests.get(argus_health_url, timeout=1.5)
            argus_online = r.status_code == 200
        except requests.RequestException:
            argus_online = False

        with _events_lock:
            events_copy = list(_system_events)

        status_data = {
            "status": "healthy",
            "scale": {
                "port": SCALE_SERIAL_PORT,
                "baudrate": SCALE_BAUD_RATE,
                "state": get_scale_state().name,
                "current_weight": 0.0,
            },
            "argus": {
                "url": target_argus_url,
                "online": argus_online,
            },
            "supabase": {
                "center_id": config.supabase_center_id,
                "authenticated": bool(auth_state.auth_token),
            },
            "cameras": {
                "cam1_url": ANPR_CAMERA_URL,
                "auxiliary_urls": AUXILIARY_CAMERA_URLS,
            },
            "config": {
                "wifi_ssid": config.wifi_ssid,
                "operator_email": config.supabase_email,
                "min_weight": config.supabase_weight_threshold,
            },
            "events": events_copy,
        }
        self._send_json_response(status_data)

    def _handle_post_wifi(self):
        content_len = int(self.headers.get("Content-Length", 0))
        post_body = self.rfile.read(content_len)

        try:
            data = json.loads(post_body.decode("utf-8"))
            ssid = data.get("ssid", "").strip()
            password = data.get("password", "")

            if not ssid:
                self._send_json_response({"success": False, "error": "SSID cannot be empty"}, 400)
                return

            config.update_wifi_credentials(ssid, password)
            record_system_event("CONFIG", f"Wi-Fi credentials updated for SSID: '{ssid}'")

            self._send_json_response({
                "success": True,
                "message": f"Wi-Fi SSID '{ssid}' successfully saved to config.json.",
            })
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            self._send_json_response({"success": False, "error": f"Invalid JSON payload: {e}"}, 400)

    def _handle_post_wifi_clear(self):
        config.clear_wifi_credentials()
        record_system_event("CONFIG", "Wi-Fi credentials cleared from config.json")
        self._send_json_response({
            "success": True,
            "message": "Wi-Fi credentials cleared from config.json.",
        })


class FallbackWebServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 8080):
        self.host = host
        self.port = port
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._is_running = False

    def start(self):
        if self._is_running:
            return

        try:
            self._server = ThreadingHTTPServer((self.host, self.port), FallbackHTTPRequestHandler)
            self._is_running = True
            self._thread = threading.Thread(
                target=self._server.serve_forever,
                name="FallbackWebServer",
                daemon=True,
            )
            self._thread.start()
            logger.info(f"[WebServer] Fallback diagnostics web server running at http://{self.host}:{self.port}")
            record_system_event("SYSTEM", f"Fallback web server started on port {self.port}")
        except OSError as e:
            logger.error(f"[WebServer] Failed to bind fallback web server on port {self.port}: {e}")

    def stop(self):
        if not self._is_running or not self._server:
            return

        self._is_running = False
        try:
            self._server.shutdown()
            self._server.server_close()
            logger.info("[WebServer] Fallback web server stopped.")
        except OSError as e:
            logger.debug(f"[WebServer] Error stopping web server: {e}")


# Module-level singleton
web_server = FallbackWebServer()


def start_web_server(host: str = "0.0.0.0", port: int = 8080):
    web_server.host = host
    web_server.port = port
    web_server.start()


def stop_web_server():
    web_server.stop()

