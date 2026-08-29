import requests
import logging
from .supabase_client import SUPABASE_BASE_URL, SUPABASE_ANON_KEY, auth_state
from ..config.config_manager import config

logger = logging.getLogger(__name__)

def login_to_supabase() -> bool:
    if not config.supabase_email or not config.supabase_password:
        logger.warning("[Auth] Missing email or password in configuration.")
        return False

    login_url = f"{SUPABASE_BASE_URL}/auth/v1/token?grant_type=password"
    logger.info("[Auth] Attempting login to Supabase...")

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
    except Exception as e:
        logger.error(f"[Auth] HTTP login exception: {e}")

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
    except Exception as e:
        logger.error(f"[Profile] Exception querying table '{table_name}': {e}")

    return False

def fetch_profile_id() -> bool:
    if not auth_state.auth_token:
        return False
    return _query_table_for_profile_id("profiles") or _query_table_for_profile_id("operators")
