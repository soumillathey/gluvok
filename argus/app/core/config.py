import os
from importlib.metadata import PackageNotFoundError, version

from pydantic_settings import BaseSettings, SettingsConfigDict

from app.schemas.plate import ProviderEnum

# Resolve the version from the installed package metadata so it stays in sync
# with pyproject.toml automatically. Falls back to "dev" when running from an
# editable install that hasn't been built (e.g. `uv run` without a prior build).
try:
    _PKG_VERSION = version("argus")
except PackageNotFoundError:
    _PKG_VERSION = "dev"


class Settings(BaseSettings):
    PROJECT_NAME: str = "Argus ANPR Microservice"
    VERSION: str = _PKG_VERSION

    # API Keys & Endpoints
    PLATE_RECOGNIZER_TOKEN: str = ""
    LLAMA_API_KEY: str = ""
    NEMOTRON_API_KEY: str = ""
    NVIDIA_INVOKE_URL: str = "https://integrate.api.nvidia.com/v1/chat/completions"
    DEFAULT_PROVIDER: ProviderEnum = ProviderEnum.DOCLING

    # YOLO Model Settings
    YOLO_MODEL_NAME: str = "yolo11n.pt"
    YOLO_CONFIG_DIR: str = ".cache/ultralytics"
    HUMAN_CONF_THRESH: float = 0.30
    VEHICLE_CONF_THRESH: float = 0.35

    # Outbound HTTP timeouts (seconds). Never leave these unset: a provider that
    # accepts the connection and then stalls will otherwise hang the worker
    # thread indefinitely.
    HTTP_CONNECT_TIMEOUT: float = 3.0
    HTTP_READ_TIMEOUT: float = 10.0

    # Upload limits
    MAX_UPLOAD_BYTES: int = 8 * 1024 * 1024  # reject the request body above this
    MAX_IMAGE_PIXELS: int = 50_000_000  # decompression-bomb guard (w * h)
    MAX_IMAGE_EDGE_PX: int = 1920  # downscale longest edge before inference

    # Pre-screening Rejection Policies
    REJECT_ON_HUMAN_DETECTED: bool = False
    REJECT_ON_MULTIPLE_VEHICLES: bool = False
    REJECT_ON_NO_VEHICLE: bool = False

    # Work bounds (fixed upper bound on vehicle boxes evaluated).
    # Boxes are area-sorted, so the largest few are the only plausible candidates.
    MAX_VEHICLE_BOXES: int = 5

    # OCR line processing upper bound
    MAX_OCR_LINES: int = 500

    # Server & CORS Settings
    SERVER_HOST: str = "0.0.0.0"
    SERVER_PORT: int = 8000
    CORS_ORIGINS: list[str] = ["*"]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: list[str] = ["*"]
    CORS_ALLOW_HEADERS: list[str] = ["*"]
    DOCS_URL: str | None = "/docs"
    REDOC_URL: str | None = "/redoc"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()

# Ultralytics resolves YOLO_CONFIG_DIR relative to the current working
# directory and appends its own "Ultralytics" subfolder. A relative path like
# ".cache/ultralytics" is not writable when the CWD differs (e.g. running from
# a service manager or a different shell), which makes ultralytics silently
# fall back to /tmp/Ultralytics. Resolve to an absolute path and create the
# directory up front so the configured location is always writable.
_yolo_config_dir = os.path.abspath(settings.YOLO_CONFIG_DIR)
os.makedirs(_yolo_config_dir, exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = _yolo_config_dir
