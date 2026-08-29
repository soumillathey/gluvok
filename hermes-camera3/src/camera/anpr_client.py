"""
anpr_client.py
ANPR Server communication client and plate voting algorithm.
Sends captured Camera 1 frames to the ANPR server and calculates the highest frequency plate.
"""

import logging
import requests
from typing import Optional, List
from collections import Counter

from ..config.camera_config import ANPR_SERVER_URL, ANPR_SERVER_TIMEOUT
from ..config.config_manager import config

logger = logging.getLogger(__name__)


def send_frame_to_anpr_server(
    image_bytes: bytes,
    server_url: Optional[str] = None,
    timeout: float = ANPR_SERVER_TIMEOUT,
) -> Optional[str]:
    """
    Sends raw JPEG image bytes to the Argus ANPR FastAPI microservice (/recognize).
    Returns the detected plate text string or None if unreadable/rejected/error.
    """
    if not image_bytes:
        return None

    # Resolve target URL: explicit arg > config.json override > camera_config default
    target_url = server_url or config.anpr_server_url or ANPR_SERVER_URL

    try:
        # Argus FastAPI expects the image under the multipart 'file' field
        files = {"file": ("frame.jpg", image_bytes, "image/jpeg")}
        response = requests.post(target_url, files=files, timeout=timeout)

        if response.status_code in [200, 201]:
            try:
                data = response.json()
                if isinstance(data, dict):
                    # 1. Handle Argus RecognitionResponse schema (list of PlateResult objects)
                    results = data.get("results")
                    if isinstance(results, list):
                        for item in results:
                            if isinstance(item, dict):
                                plate_val = item.get("plate")
                                if plate_val and isinstance(plate_val, str) and plate_val.strip() and plate_val.strip().upper() != "N/A":
                                    plate_clean = plate_val.strip().upper()
                                    exec_time = data.get("execution_time_ms", "N/A")
                                    provider = data.get("provider", "unknown")
                                    vtype = data.get("vehicle_type") or "vehicle"
                                    logger.info(
                                        f"[ANPR] Argus recognized plate: '{plate_clean}' "
                                        f"({vtype}, provider={provider}, {exec_time}ms)"
                                    )
                                    return plate_clean

                    # 2. Check if frame was rejected during pre-screening
                    if data.get("rejected"):
                        status_msg = data.get("status_message", "Pre-screening rejected frame")
                        logger.info(f"[ANPR] Argus pre-screening rejected frame: {status_msg} (status: {data.get('status')})")
                        return None

                    # 3. Fallback for flat JSON responses
                    for key in ("plate", "number_plate", "plate_number", "text", "result"):
                        flat_val = data.get(key)
                        if flat_val and isinstance(flat_val, str) and flat_val.strip() and flat_val.strip().upper() != "N/A":
                            plate_clean = flat_val.strip().upper()
                            logger.info(f"[ANPR] Server returned plate: '{plate_clean}' (key='{key}')")
                            return plate_clean

                elif isinstance(data, str) and data.strip():
                    return data.strip().upper()

            except Exception as parse_err:
                # If plain text response
                text = response.text.strip().upper()
                if text and text != "N/A":
                    logger.info(f"[ANPR] Server returned text response: '{text}'")
                    return text
                logger.warning(f"[ANPR] Failed to parse server response: {parse_err}")

        logger.warning(f"[ANPR] Server POST status {response.status_code}: {response.text[:120]}")
        return None
    except requests.exceptions.Timeout:
        logger.warning(f"[ANPR] Timeout ({timeout}s) contacting ANPR server at {target_url}")
        return None
    except requests.exceptions.ConnectionError:
        logger.warning(f"[ANPR] Failed to connect to ANPR server at {target_url}. Is Argus running?")
        return None
    except Exception as e:
        logger.error(f"[ANPR] Exception submitting frame to ANPR server ({target_url}): {e}")
        return None


def get_highest_frequency_plate(plate_list: List[str]) -> str:
    """
    Analyzes a list of plate reading strings collected during a session and returns
    the string with the highest frequency.
    Returns 'UNKNOWN_PLATE' if no valid plate was recognized.
    """
    valid_plates = [p.strip().upper() for p in plate_list if p and isinstance(p, str) and p.strip()]

    if not valid_plates:
        logger.warning("[ANPR Voting] No valid plates recorded during session.")
        return "UNKNOWN_PLATE"

    counter = Counter(valid_plates)
    most_common_plate, count = counter.most_common(1)[0]
    total_samples = len(valid_plates)

    logger.info(
        f"[ANPR Voting] Winner: '{most_common_plate}' "
        f"(Frequency: {count}/{total_samples}, All candidates: {dict(counter)})"
    )
    return most_common_plate
