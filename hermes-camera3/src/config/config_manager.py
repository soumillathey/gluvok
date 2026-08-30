import json
import logging
import os

logger = logging.getLogger(__name__)

CONFIG_FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.json")

class ConfigManager:
    def __init__(self, file_path=CONFIG_FILE_PATH):
        self.file_path = file_path
        self.wifi_ssid = ""
        self.wifi_password = ""
        self.supabase_center_id = 1
        self.supabase_weight_threshold = 50.0
        self.supabase_email = ""
        self.supabase_password = ""
        self.supabase_profile_id = -1
        self.anpr_server_url = ""
        self.load_settings()

    def load_settings(self):
        if not os.path.exists(self.file_path):
            logger.info(f"[Config] Config file '{self.file_path}' not found. Initializing with defaults.")
            self.save_settings(
                ssid="",
                password="",
                center_id=1,
                min_weight=50.0,
                sb_email="",
                sb_password="",
                anpr_url=""
            )
            return

        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.wifi_ssid = data.get("ssid", "")
            self.wifi_password = data.get("password", "")
            self.supabase_center_id = int(data.get("center_id", 1))
            self.supabase_weight_threshold = float(data.get("min_weight", 50.0))
            self.supabase_email = data.get("sb_email", "")
            self.supabase_password = data.get("sb_pass", "")
            self.supabase_profile_id = int(data.get("profile_id", -1))
            self.anpr_server_url = data.get("anpr_server_url", "")

            logger.info("Configurations loaded from JSON storage:")
            logger.info(f" -> SSID: {self.wifi_ssid}")
            logger.info(f" -> Operator Email: {self.supabase_email}")
            logger.info(f" -> Center ID: {self.supabase_center_id}")
            logger.info(f" -> Min Weight Threshold: {self.supabase_weight_threshold:.1f}")
            if self.anpr_server_url:
                logger.info(f" -> ANPR Server URL Override: {self.anpr_server_url}")
        except (OSError, json.JSONDecodeError, ValueError) as e:
            logger.error(f"[Config] Error loading settings: {e}")

    def save_settings(self, ssid, password, center_id, min_weight, sb_email, sb_password, anpr_url=None):
        self.wifi_ssid = ssid
        self.wifi_password = password
        self.supabase_center_id = int(center_id)
        self.supabase_weight_threshold = float(min_weight)
        self.supabase_email = sb_email
        self.supabase_password = sb_password
        if anpr_url is not None:
            self.anpr_server_url = anpr_url

        data = {
            "ssid": self.wifi_ssid,
            "password": self.wifi_password,
            "center_id": self.supabase_center_id,
            "min_weight": self.supabase_weight_threshold,
            "sb_email": self.supabase_email,
            "sb_pass": self.supabase_password,
            "profile_id": self.supabase_profile_id,
            "anpr_server_url": self.anpr_server_url
        }

        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.info("[Config] New configurations written to persistent JSON storage.")
        except (OSError, TypeError, ValueError) as e:
            logger.error(f"[Config] Failed to save settings: {e}")

    def update_profile_id(self, profile_id):
        self.supabase_profile_id = profile_id
        self.save_settings(
            self.wifi_ssid,
            self.wifi_password,
            self.supabase_center_id,
            self.supabase_weight_threshold,
            self.supabase_email,
            self.supabase_password
        )

    def update_wifi_credentials(self, ssid: str, password: str):
        self.wifi_ssid = ssid
        self.wifi_password = password
        self.save_settings(
            ssid=self.wifi_ssid,
            password=self.wifi_password,
            center_id=self.supabase_center_id,
            min_weight=self.supabase_weight_threshold,
            sb_email=self.supabase_email,
            sb_password=self.supabase_password,
            anpr_url=self.anpr_server_url,
        )
        logger.info(f"[Config] Wi-Fi credentials updated for SSID: '{self.wifi_ssid}'.")

    def clear_wifi_credentials(self):
        self.wifi_ssid = ""
        self.wifi_password = ""
        self.save_settings(
            ssid="",
            password="",
            center_id=self.supabase_center_id,
            min_weight=self.supabase_weight_threshold,
            sb_email=self.supabase_email,
            sb_password=self.supabase_password,
            anpr_url=self.anpr_server_url,
        )
        logger.info("[Config] Wi-Fi credentials cleared.")

# Shared singleton instance matching ESP32 global settings
config = ConfigManager()

