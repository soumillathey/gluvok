"""
scale_stability.py
Weight session state machine with 10-second continuous stability detection.
Integrates with WeighbridgeSessionManager for camera captures and ANPR processing.
One transmission per session; resets only when weight returns to zero.
"""

import logging
import time
from enum import Enum

from ..camera.session_manager import session_manager
from ..config.config_manager import config

logger = logging.getLogger(__name__)

class ScaleState(Enum):
    SCALE_IDLE             = 0
    SCALE_STABILIZING      = 1
    SCALE_STABLE_RECORDED  = 2

STABILITY_TOLERANCE = 2.0    # ±2.0 kg allowed variation
STABILITY_DURATION  = 10.0   # 10 seconds continuous stability required

class ScaleStabilityMachine:
    def __init__(self):
        self.state = ScaleState.SCALE_IDLE
        self._current_stable_candidate = 0.0
        self._candidate_start_time = 0.0
        self._last_printed_weight = -9999.0

    def process_new_weight(self, parsed_weight: float):
        now = time.time()

        # Log weight if it changed significantly (≥0.1 kg)
        if abs(parsed_weight - self._last_printed_weight) >= 0.1:
            logger.info(
                f"[Scale] Parsed weight: {parsed_weight:.3f} "
                f"(Threshold: {config.supabase_weight_threshold:.1f})"
            )
            self._last_printed_weight = parsed_weight

        # Check if 10-second post-stabilization period completed and package is ready
        completed_package = session_manager.check_session_progress()
        if completed_package:
            self._trigger_upload(completed_package)

        # ── Session end: weight returned to zero ──────────────────────────────
        if parsed_weight <= 0.0:
            if self.state != ScaleState.SCALE_IDLE:
                logger.info(
                    "[Scale Session] Weight returned to zero. "
                    "Session closed. Ready for next weighing."
                )
                self.state = ScaleState.SCALE_IDLE
                self._current_stable_candidate = 0.0
                self._candidate_start_time = 0.0
                session_manager.reset_session()
            return

        # ── Lockout: only one upload per session ──────────────────────────────
        if self.state == ScaleState.SCALE_STABLE_RECORDED:
            return

        # ── Below threshold: reset ─────────────────────────────────────────────
        if parsed_weight < config.supabase_weight_threshold:
            self.state = ScaleState.SCALE_IDLE
            self._current_stable_candidate = 0.0
            self._candidate_start_time = 0.0
            session_manager.reset_session()
            return

        # ── Start stability timer & camera ANPR session on new trigger ────────
        if self.state == ScaleState.SCALE_IDLE:
            self.state = ScaleState.SCALE_STABILIZING
            self._current_stable_candidate = parsed_weight
            self._candidate_start_time = now
            session_manager.start_session()
            return

        # ── Evaluate 10-second stability window ───────────────────────────────
        if abs(parsed_weight - self._current_stable_candidate) <= STABILITY_TOLERANCE:
            elapsed = now - self._candidate_start_time
            if elapsed >= STABILITY_DURATION:
                logger.info(
                    f"[Scale Session] Stable weight confirmed (10s): "
                    f"{self._current_stable_candidate:.3f} kg. Triggering auxiliary cameras..."
                )
                self.state = ScaleState.SCALE_STABLE_RECORDED
                session_manager.on_weight_stabilized(self._current_stable_candidate)
        else:
            # Weight shifted — reset candidate and timer
            logger.debug(
                f"[Scale] Weight shifted from {self._current_stable_candidate:.3f} "
                f"to {parsed_weight:.3f}. Resetting stability timer."
            )
            self._current_stable_candidate = parsed_weight
            self._candidate_start_time = now

    def _trigger_upload(self, session_package: dict):
        # Import here to avoid circular imports
        from ..network.supabase_post import post_to_supabase
        post_to_supabase(session_package)

    def reset(self):
        """Manually reset the state machine."""
        self.state = ScaleState.SCALE_IDLE
        self._current_stable_candidate = 0.0
        self._candidate_start_time = 0.0
        self._last_printed_weight = -9999.0
        session_manager.reset_session()

# Module-level singleton
scale_state_machine = ScaleStabilityMachine()

# Convenience function used by scale_uart.py
def process_new_weight(weight: float):
    scale_state_machine.process_new_weight(weight)

# Expose current state
def get_scale_state() -> ScaleState:
    return scale_state_machine.state

