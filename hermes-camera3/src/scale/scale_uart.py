"""
scale_uart.py
UART serial stream reader and line parser for the weighing scale indicator.
Handles: persistent character buffering, packet terminators (\r, \n, STX, ETX),
inter-packet timeout flush (300ms), and flexible numeric extraction.
Maps to: ESP32 src/scale/scale_uart.cpp
"""

import logging
import re
import threading
import time

import serial

logger = logging.getLogger(__name__)

# Serial port configuration (Raspberry Pi UART)
SCALE_SERIAL_PORT = "/dev/ttyAMA0"  # Hardware UART (GPIO 14=TX, GPIO 15=RX); fallback: /dev/serial0, /dev/ttyS0
SCALE_BAUD_RATE = 1200
SCALE_TIMEOUT = 0.05  # 50ms non-blocking read timeout

# Inter-character timeout: flush buffer after 300ms silence (matching ESP32)
INTER_CHAR_TIMEOUT_S = 0.3
MAX_BUFFER_LEN = 256

class ScaleUARTReader:
    def __init__(self):
        self._serial = None
        self._running = False
        self._thread = None
        self._line_buffer = bytearray()
        self._last_char_time = 0.0
        self._lock = threading.Lock()

    def start(self, port: str = SCALE_SERIAL_PORT, baudrate: int = SCALE_BAUD_RATE):
        try:
            self._serial = serial.Serial(port, baudrate, timeout=SCALE_TIMEOUT)
            logger.info(f"[Scale] UART opened: {port} @ {baudrate} baud")
        except (serial.SerialException, OSError) as e:
            logger.error(f"[Scale] Failed to open UART port '{port}': {e}")
            return

        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True, name="ScaleUART")
        self._thread.start()
        logger.info("[Scale] UART reader thread started.")

    def stop(self):
        self._running = False
        if self._serial and self._serial.is_open:
            try:
                self._serial.close()
            except (serial.SerialException, OSError) as e:
                logger.debug(f"[Scale] Error closing serial port: {e}")
        logger.info("[Scale] UART reader stopped.")

    def _read_loop(self):
        logger.info("[Scale] UART reader thread listening for scale weight data...")
        while self._running:
            try:
                if self._serial and self._serial.is_open:
                    n = getattr(self._serial, 'in_waiting', 1) or 1
                    raw = self._serial.read(max(1, min(n, 64)))
                    if raw:
                        for b in raw:
                            self.handle_scale_char(b)
                    self._check_timeout_flush()
                else:
                    time.sleep(0.1)
            except (serial.SerialException, OSError, TypeError) as e:
                if not self._running:
                    break
                logger.error(f"[Scale] UART read error: {e}")
                time.sleep(0.5)

    def handle_scale_char(self, c):
        """
        Consumes a single character/byte or string/bytes into persistent buffer.
        Triggers parsing on packet terminators (CR, LF, STX, ETX) or when max buffer length is reached.
        """
        if isinstance(c, str):
            char_bytes = c.encode('utf-8', errors='ignore')
        elif isinstance(c, int):
            char_bytes = bytes([c])
        elif isinstance(c, (bytes, bytearray)):
            char_bytes = bytes(c)
        else:
            return

        for b in char_bytes:
            buf_to_parse = None
            with self._lock:
                self._line_buffer.append(b)
                self._last_char_time = time.time()

                # Terminators: CR (\r = 13), LF (\n = 10), STX (\x02 = 2), ETX (\x03 = 3)
                if b in (13, 10, 2, 3) or len(self._line_buffer) >= MAX_BUFFER_LEN:
                    buf_to_parse = bytes(self._line_buffer)
                    self._line_buffer.clear()

            if buf_to_parse:
                self._parse_raw_buffer(buf_to_parse)

    def _check_timeout_flush(self):
        """Flushes the line buffer if inter-character timeout (300ms) has elapsed."""
        buf_to_parse = None
        with self._lock:
            if self._line_buffer and (time.time() - self._last_char_time >= INTER_CHAR_TIMEOUT_S):
                buf_to_parse = bytes(self._line_buffer)
                self._line_buffer.clear()

        if buf_to_parse:
            self._parse_raw_buffer(buf_to_parse)

    def _parse_raw_buffer(self, buf: bytes):
        """Extracts weight numeric value from raw packet buffer and notifies stability logic."""
        if not buf:
            return

        # 1. Primary indicator protocol check: 3-6 digits followed by MN (e.g. 26500MN)
        m = re.search(rb"([0-9]{3,6})MN", buf)
        if m:
            try:
                val = float(int(m.group(1)))
                handle_scale_char_processed(val)
                return
            except ValueError:
                pass

        # 2. Fallback flexible numeric extraction (supports signed float/int e.g. +05000.5 or -12.5)
        text = buf.decode('utf-8', errors='ignore')
        m_flex = re.search(r"([-+]?\d+(?:\.\d+)?)", text)
        if m_flex:
            try:
                val = float(m_flex.group(1))
                handle_scale_char_processed(val)
                return
            except ValueError:
                pass

# Module-level singleton and callback bridge
_uart_reader = ScaleUARTReader()

def handle_scale_char(c):
    """Called per-character/byte from external sources (for testing or alternate serial readers)."""
    _uart_reader.handle_scale_char(c)

def handle_scale_char_processed(weight: float):
    """Bridge to stability state machine — called after a weight value is extracted."""
    logger.info(f"[Scale] Parsed weight: {weight:.3f} kg")
    from ..scale.scale_stability import process_new_weight
    process_new_weight(weight)

def get_uart_reader() -> ScaleUARTReader:
    return _uart_reader

