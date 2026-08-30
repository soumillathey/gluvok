import base64
import logging
from typing import Any

import requests

from ..config.config_manager import config
from .supabase_auth import login_to_supabase
from .supabase_client import SUPABASE_ANON_KEY, SUPABASE_BASE_URL, auth_state

logger = logging.getLogger(__name__)

SUPABASE_TABLE = "weighments"

def post_to_supabase(session_payload: float | dict[str, Any], is_retry: bool = False):
    if not auth_state.auth_token and not login_to_supabase():
        logger.warning("[Supabase] POST aborted: Login failed.")
        return

    full_url = f"{SUPABASE_BASE_URL}/rest/v1/{SUPABASE_TABLE}"
    logger.info(f"[Supabase] Starting POST to: {full_url}")

    if isinstance(session_payload, dict):
        weight_value = session_payload.get("weight", 0.0)
        vehicle_number = session_payload.get("anpr_plate", "UNKNOWN_PLATE")
        session_id = session_payload.get("session_id", "")

        # Base64 encode images in RAM for JSON payload
        cam1_bytes = session_payload.get("cam1_final_image")
        cam1_b64 = base64.b64encode(cam1_bytes).decode("utf-8") if cam1_bytes else None

        aux_images = session_payload.get("auxiliary_images", {})
        aux_b64_dict = {}
        for cam_idx, img_bytes in aux_images.items():
            if img_bytes:
                aux_b64_dict[f"cam_{cam_idx}"] = base64.b64encode(img_bytes).decode("utf-8")

        payload = {
            "session_id": session_id,
            "weight": round(weight_value, 3),
            "vehicle_number": vehicle_number,
            "rate_id": 1,
            "center_id": config.supabase_center_id,
            "customer_id": 1,
            "cam1_image_base64": cam1_b64,
            "auxiliary_images_base64": aux_b64_dict,
        }
    else:
        weight_value = float(session_payload)
        payload = {
            "weight": round(weight_value, 3),
            "vehicle_number": "MH12AB1234",
            "rate_id": 1,
            "center_id": config.supabase_center_id,
            "customer_id": 1
        }

    if config.supabase_profile_id != -1:
        payload["profile_id"] = config.supabase_profile_id

    logger.info(
        f"[Supabase] Payload prepared for vehicle '{payload.get('vehicle_number')}' "
        f"weight={payload.get('weight')} kg (session: {payload.get('session_id', 'N/A')})"
    )

    headers = {
        "Content-Type": "application/json",
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {auth_state.auth_token}",
        "Prefer": "return=minimal",
        "Connection": "close"
    }

    try:
        response = requests.post(full_url, json=payload, headers=headers, timeout=15)
        if response.status_code in [200, 201]:
            logger.info(f"[Supabase] POST success. Response Code: {response.status_code}")
        elif response.status_code in [401, 403] and not is_retry:
            logger.warning("[Supabase] Token expired or invalid. Refreshing token and retrying...")
            auth_state.auth_token = ""
            if login_to_supabase():
                post_to_supabase(session_payload, is_retry=True)
            else:
                try:
                    from ..web.server import record_error_event, record_system_event
                    record_error_event("CLOUD_AUTH_FAILED", "Token refresh failed")
                    record_system_event("CLOUD", "Cloud token refresh failed.")
                except (ImportError, AttributeError):
                    pass
        else:
            logger.error(f"[Supabase] POST error {response.status_code}: {response.text}")
            try:
                from ..web.server import record_error_event, record_system_event
                record_error_event("CLOUD_UPLOAD_ERROR", f"HTTP {response.status_code}")
                record_system_event("CLOUD", f"Cloud POST failed: HTTP {response.status_code}")
            except (ImportError, AttributeError):
                pass
    except (requests.RequestException, ValueError, KeyError) as e:
        logger.error(f"[Supabase] HTTP POST exception: {e}")
        try:
            from ..web.server import record_error_event, record_system_event
            record_error_event("CLOUD_UPLOAD_ERROR", str(e))
            record_system_event("CLOUD", f"Cloud POST exception: {e}")
        except (ImportError, AttributeError):
            pass
    finally:

        # Explicitly release image byte buffers and base64 payloads from RAM
        if isinstance(session_payload, dict):
            session_payload.clear()
        payload.clear()


