"""
wifi_manager.py
Manages Raspberry Pi Wi-Fi connectivity and automatic emergency Access Point (Hotspot) fallback.
Uses NetworkManager (`nmcli`) to detect connection drops and spin up the 'Gluvok-Setup' hotspot.
"""

import logging
import shutil
import subprocess
import threading

logger = logging.getLogger(__name__)

HOTSPOT_CON_NAME = "Gluvok-Hotspot"
DEFAULT_HOTSPOT_SSID = "Gluvok-Setup"
DEFAULT_HOTSPOT_PASS = "gluvok1234"

_hotspot_active = False
_watchdog_thread: threading.Thread | None = None
_watchdog_stop_event = threading.Event()
_wifi_lock = threading.Lock()


def is_nmcli_available() -> bool:
    """Checks if nmcli (NetworkManager) is installed on the system."""
    return shutil.which("nmcli") is not None


def is_wifi_connected() -> bool:
    """
    Checks if wlan0 is connected to an active Wi-Fi network (not in AP/hotspot mode).
    Returns True if connected to an external Wi-Fi SSID, False otherwise.
    """
    if not is_nmcli_available():
        # Fallback check for non-Linux or mock environments
        return False

    try:
        # Query active NetworkManager Wi-Fi connections
        res = subprocess.run(
            ["nmcli", "-t", "-f", "TYPE,STATE,CONNECTION", "dev"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if res.returncode != 0:
            return False

        for line in res.stdout.strip().splitlines():
            parts = line.split(":")
            if len(parts) >= 3 and parts[0] == "wifi" and parts[1] == "connected":
                con_name = parts[2]
                # If connected but it's our own hotspot, it's not a normal external Wi-Fi
                if con_name != HOTSPOT_CON_NAME:
                    return True
        return False
    except (subprocess.SubprocessError, OSError) as e:
        logger.debug(f"[WiFi] Error checking Wi-Fi status: {e}")
        return False


def is_hotspot_active() -> bool:
    """Returns True if the emergency hotspot is currently broadcasting."""
    return _hotspot_active



def start_emergency_hotspot(
    ssid: str = DEFAULT_HOTSPOT_SSID,
    password: str = DEFAULT_HOTSPOT_PASS,
) -> bool:
    """
    Starts the emergency Wi-Fi Access Point (Hotspot) on wlan0 using nmcli.
    Broadcasts 'Gluvok-Setup' with default IP 10.42.0.1.
    """
    global _hotspot_active
    with _wifi_lock:
        if _hotspot_active:
            return True

        if not is_nmcli_available():
            logger.info("[WiFi] nmcli not available. Simulating hotspot start in current environment.")
            _hotspot_active = True
            return True

        try:
            # Delete existing hotspot profile if present to ensure clean state
            subprocess.run(["nmcli", "connection", "delete", HOTSPOT_CON_NAME], capture_output=True, timeout=5, check=False)

            # Create and configure the AP hotspot connection
            cmd_create = [
                "nmcli", "connection", "add",
                "type", "wifi",
                "ifname", "wlan0",
                "con-name", HOTSPOT_CON_NAME,
                "autoconnect", "false",
                "ssid", ssid,
            ]
            subprocess.run(cmd_create, capture_output=True, timeout=5, check=True)

            cmd_modify = [
                "nmcli", "connection", "modify", HOTSPOT_CON_NAME,
                "802-11-wireless.mode", "ap",
                "802-11-wireless.band", "bg",
                "ipv4.method", "shared",
                "wifi-sec.key-mgmt", "wpa-psk",
                "wifi-sec.psk", password,
            ]
            subprocess.run(cmd_modify, capture_output=True, timeout=5, check=True)

            # Bring up the hotspot
            subprocess.run(["nmcli", "connection", "up", HOTSPOT_CON_NAME], capture_output=True, timeout=10, check=True)
            _hotspot_active = True
            logger.info(
                f"[WiFi] Emergency Access Point active! SSID: '{ssid}' | "
                f"Password: '{password}' | Web Console: http://10.42.0.1:8080"
            )
            return True
        except (subprocess.SubprocessError, OSError) as e:
            logger.error(f"[WiFi] Failed to start emergency hotspot: {e}")
            return False


def stop_emergency_hotspot() -> bool:
    """Tears down the emergency Access Point on wlan0."""
    global _hotspot_active
    with _wifi_lock:
        if not _hotspot_active:
            return True

        if not is_nmcli_available():
            _hotspot_active = False
            return True

        try:
            subprocess.run(["nmcli", "connection", "down", HOTSPOT_CON_NAME], capture_output=True, timeout=5, check=False)
            _hotspot_active = False
            logger.info("[WiFi] Emergency Access Point stopped.")
            return True
        except (subprocess.SubprocessError, OSError) as e:
            logger.error(f"[WiFi] Error stopping emergency hotspot: {e}")
            return False


def connect_to_wifi(ssid: str, password: str) -> tuple[bool, str]:
    """
    Attempts to connect wlan0 to the specified Wi-Fi network using nmcli.
    If connected successfully, tears down the emergency hotspot.
    """
    if not ssid:
        return False, "SSID cannot be empty."

    if not is_nmcli_available():
        logger.info(f"[WiFi] nmcli not available. Mock connected to '{ssid}'.")
        stop_emergency_hotspot()
        return True, f"Connected to '{ssid}' (simulated)."

    logger.info(f"[WiFi] Attempting connection to Wi-Fi SSID: '{ssid}'...")
    try:
        # Stop hotspot temporarily to free up the wireless device
        if is_hotspot_active():
            subprocess.run(["nmcli", "connection", "down", HOTSPOT_CON_NAME], capture_output=True, timeout=5, check=False)

        cmd = ["nmcli", "dev", "wifi", "connect", ssid]
        if password:
            cmd.extend(["password", password])

        res = subprocess.run(cmd, capture_output=True, text=True, timeout=20, check=False)

        if res.returncode == 0:
            logger.info(f"[WiFi] Successfully connected to Wi-Fi: '{ssid}'!")
            stop_emergency_hotspot()
            return True, f"Successfully connected to Wi-Fi: '{ssid}'."
        else:
            err_msg = res.stderr.strip() or res.stdout.strip() or "Connection failed"
            logger.warning(f"[WiFi] Failed to connect to '{ssid}': {err_msg}")
            # Re-engage emergency hotspot so technician does not lose connection
            start_emergency_hotspot()
            return False, f"Failed to connect to '{ssid}': {err_msg}"
    except (subprocess.SubprocessError, OSError) as e:
        logger.error(f"[WiFi] Subprocess error connecting to '{ssid}': {e}")
        start_emergency_hotspot()
        return False, f"System error connecting to Wi-Fi: {e}"


def _watchdog_loop(interval: float):
    """Monitors Wi-Fi connection and triggers emergency hotspot if disconnected."""
    logger.info(f"[WiFi Watchdog] Started network monitoring (check interval: {interval}s).")
    while not _watchdog_stop_event.is_set():
        if not is_wifi_connected():
            if not is_hotspot_active():
                logger.warning("[WiFi Watchdog] No active Wi-Fi connection detected! Starting emergency hotspot...")
                start_emergency_hotspot()
        else:
            if is_hotspot_active():
                logger.info("[WiFi Watchdog] Wi-Fi connection restored. Stopping emergency hotspot.")
                stop_emergency_hotspot()

        _watchdog_stop_event.wait(interval)

    logger.info("[WiFi Watchdog] Stopped network monitoring.")


def start_wifi_watchdog(interval: float = 30.0):
    """Starts the background Wi-Fi monitoring thread."""
    global _watchdog_thread
    if _watchdog_thread and _watchdog_thread.is_alive():
        return

    _watchdog_stop_event.clear()
    _watchdog_thread = threading.Thread(
        target=_watchdog_loop,
        args=(interval,),
        name="WiFiWatchdog",
        daemon=True,
    )
    _watchdog_thread.start()


def stop_wifi_watchdog():
    """Stops the background Wi-Fi monitoring thread."""
    _watchdog_stop_event.set()
