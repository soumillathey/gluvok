"""
session_manager.py
Manages weighbridge session lifecycle, Camera 1 ANPR capture loop thread,
auxiliary camera snapshots, 10-second post-stabilization timing, and data packaging.
"""

import logging
import threading
import time
import uuid
from enum import Enum
from typing import Any

import requests

from ..config.camera_config import (
    ANPR_CAMERA_URL,
    ANPR_CAPTURE_INTERVAL,
    POST_STABILITY_DURATION,
)
from .anpr_client import get_highest_frequency_plate, send_frame_to_anpr_server
from .camera_manager import capture_auxiliary_snapshots, fetch_image_bytes

logger = logging.getLogger(__name__)


class SessionPhase(Enum):
    PHASE_IDLE = 0
    PHASE_STABILIZING = 1          # Weight active, 2s Cam 1 ANPR running
    PHASE_POST_STABILITY = 2       # Weight stable, Aux cams captured, +10s timer running
    PHASE_COMPLETED = 3            # Session finished & transmitted, awaiting weight return to 0


class WeighbridgeSessionManager:
    def __init__(self):
        self.phase = SessionPhase.PHASE_IDLE
        self.session_id: str | None = None
        self.stable_weight: float = 0.0

        self._cam1_frames: list[bytes] = []
        self._cam1_plates: list[str] = []
        self._auxiliary_images: dict[int, bytes | None] = {}

        self._anpr_thread: threading.Thread | None = None
        self._stop_anpr_event = threading.Event()

        self._post_stability_start_time: float = 0.0
        self._lock = threading.Lock()

    def start_session(self):
        """Called when scale weight exceeds threshold (start of weighment session)."""
        with self._lock:
            if self.phase != SessionPhase.PHASE_IDLE:
                return

            self.session_id = f"SESS_{int(time.time())}_{uuid.uuid4().hex[:6]}"
            self.phase = SessionPhase.PHASE_STABILIZING
            self.stable_weight = 0.0
            self._cam1_frames.clear()
            self._cam1_plates.clear()
            self._auxiliary_images.clear()
            self._stop_anpr_event.clear()

            logger.info(f"[Session] Started new weighbridge session: {self.session_id}")

            # Launch Camera 1 ANPR 2-second capture loop thread
            self._anpr_thread = threading.Thread(
                target=self._anpr_loop,
                name=f"ANPRLoop_{self.session_id}",
                daemon=True
            )
            self._anpr_thread.start()

    def _anpr_loop(self):
        """Background thread executing 2-second Camera 1 capture & ANPR requests."""
        logger.info(f"[Session {self.session_id}] Camera 1 ANPR capture loop started.")
        while not self._stop_anpr_event.is_set():
            loop_start = time.time()

            try:
                img_bytes = fetch_image_bytes(ANPR_CAMERA_URL)
                if img_bytes:
                    with self._lock:
                        # Keep only recent frames in RAM to prevent memory bloat
                        if len(self._cam1_frames) >= 5:
                            self._cam1_frames.pop(0)
                        self._cam1_frames.append(img_bytes)

                    plate = send_frame_to_anpr_server(img_bytes)
                    if plate:
                        with self._lock:
                            self._cam1_plates.append(plate)
            except (requests.RequestException, OSError, ValueError, RuntimeError) as e:
                logger.error(f"[Session {self.session_id}] Exception in ANPR loop iteration: {e}")

            # Sleep remaining time to maintain 2.0s interval
            elapsed = time.time() - loop_start
            sleep_time = max(0.1, ANPR_CAPTURE_INTERVAL - elapsed)
            time.sleep(sleep_time)

        logger.info(f"[Session {self.session_id}] Camera 1 ANPR capture loop stopped.")

    def on_weight_stabilized(self, weight: float):
        """Called when scale stability machine confirms 10s weight stability."""
        with self._lock:
            if self.phase != SessionPhase.PHASE_STABILIZING:
                return

            self.phase = SessionPhase.PHASE_POST_STABILITY
            self.stable_weight = weight
            self._post_stability_start_time = time.time()
            logger.info(
                f"[Session {self.session_id}] Weight stabilized at {weight:.3f} kg. "
                f"Triggering auxiliary camera snapshots and starting +10s countdown..."
            )

        # Concurrently capture auxiliary overview cameras (2 ... N)
        aux_images = capture_auxiliary_snapshots()
        with self._lock:
            self._auxiliary_images = aux_images

    def check_session_progress(self) -> dict[str, Any] | None:
        """
        Periodically called in loop. Checks if 10-second post-stabilization timer expired.
        If expired, finalizes session, stops ANPR loop, and returns full session package.
        """
        with self._lock:
            if self.phase != SessionPhase.PHASE_POST_STABILITY:
                return None

            elapsed = time.time() - self._post_stability_start_time
            if elapsed < POST_STABILITY_DURATION:
                return None  # Still waiting out the 10-second post-stabilization period

            logger.info(
                f"[Session {self.session_id}] Post-stabilization 10s completed. "
                f"Finalizing session package..."
            )
            self.phase = SessionPhase.PHASE_COMPLETED
            self._stop_anpr_event.set()

        # Build final session package
        final_package = self._finalize_session_package()
        return final_package

    def _finalize_session_package(self) -> dict[str, Any]:
        """Assembles final session dictionary data."""
        with self._lock:
            final_anpr_plate = get_highest_frequency_plate(self._cam1_plates)
            last_cam1_image = self._cam1_frames[-1] if self._cam1_frames else None

            package = {
                "session_id": self.session_id,
                "weight": round(self.stable_weight, 3),
                "anpr_plate": final_anpr_plate,
                "cam1_final_image": last_cam1_image,
                "auxiliary_images": self._auxiliary_images,
                "total_anpr_samples": len(self._cam1_plates),
                "total_cam1_frames": len(self._cam1_frames),
                "timestamp": time.time(),
            }

            logger.info(
                f"[Session {self.session_id}] Package finalized: "
                f"Plate='{final_anpr_plate}', Weight={self.stable_weight:.3f} kg, "
                f"Cam1 Frames={len(self._cam1_frames)}, Aux Cams={len(self._auxiliary_images)}"
            )
            return package

    def reset_session(self):
        """Resets session manager state when weight returns to zero (scale idle)."""
        with self._lock:
            if self.phase == SessionPhase.PHASE_IDLE:
                return

            logger.info(f"[Session {self.session_id}] Weight zeroed. Resetting session manager.")
            self._stop_anpr_event.set()
            self.phase = SessionPhase.PHASE_IDLE
            self.session_id = None
            self.stable_weight = 0.0
            self._cam1_frames.clear()
            self._cam1_plates.clear()
            self._auxiliary_images.clear()


# Module-level singleton
session_manager = WeighbridgeSessionManager()
