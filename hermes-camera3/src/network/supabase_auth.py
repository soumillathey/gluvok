import logging

import requests

from ..config.config_manager import config
from .supabase_client import SUPABASE_ANON_KEY, SUPABASE_BASE_URL, auth_state

logger = logging.getLogger(__name__)

def login_to_supabase() -> bool:
    if not config.supabase_email or not config.supabase_password:
        logger.warning("[Auth] Missing email or password in configuration.")
        try:
            from ..web.server import record_error_event, record_system_event
            record_error_event("CLOUD_AUTH_FAILED", "Missing email or password in configuration.")
            record_system_event("CLOUD", "Auth failed: Missing credentials in config.json")
        except (ImportError, AttributeError):
            pass
        return False

    login_url = f"{SUPABASE_BASE_URL}/auth/v1/token?grant_type=password"
    logger.info("[Auth] Attempting login to Cloud Backend...")

    headers = {
        "Content-Type": "application/json",
        "apikey": SUPABASE_ANON_KEY,
        "Connection": "close"
    }
    payload = {
        "email": config.supabase_email,
        "password": config.supabase_password
    }

    try:
        response = requests.post(login_url, json=payload, headers=headers, timeout=15)
        if response.status_code in [200, 201]:
            data = response.json()
            token = data.get("access_token")
            if token:
                auth_state.auth_token = token
                logger.info("[Auth] Login successful. Access Token acquired.")
                fetch_profile_id()
                return True
        logger.error(f"[Auth] Login failed with status code: {response.status_code}, response: {response.text}")
        try:
            from ..web.server import record_error_event, record_system_event
            record_error_event("CLOUD_AUTH_FAILED", f"Status: {response.status_code}")
            record_system_event("CLOUD", f"Cloud login failed: HTTP {response.status_code}")
        except (ImportError, AttributeError):
            pass
    except (requests.RequestException, ValueError, KeyError) as e:
        logger.error(f"[Auth] HTTP login exception: {e}")
        try:
            from ..web.server import record_error_event, record_system_event
            record_error_event("CLOUD_AUTH_FAILED", str(e))
            record_system_event("CLOUD", f"Cloud login network exception: {e}")
        except (ImportError, AttributeError):
            pass

    return False


def _query_table_for_profile_id(table_name: str) -> bool:
    url = f"{SUPABASE_BASE_URL}/rest/v1/{table_name}?select=id"
    headers = {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {auth_state.auth_token}",
        "Connection": "close"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) > 0 and "id" in data[0]:
                profile_id = int(data[0]["id"])
                config.update_profile_id(profile_id)
                logger.info(f"[Profile] Resolved Operator Profile ID from table '{table_name}': {profile_id}")
                return True
    except (requests.RequestException, ValueError, KeyError) as e:
        logger.error(f"[Profile] Exception querying table '{table_name}': {e}")

    return False

def fetch_profile_id() -> bool:
    if not auth_state.auth_token:
        return False
    return _query_table_for_profile_id("profiles") or _query_table_for_profile_id("operators")
