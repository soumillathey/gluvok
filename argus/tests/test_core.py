from app.core.config import settings
from app.core.exceptions import (
    ANPRServiceError,
    InvalidImageError,
    ProviderNotFoundError,
)


def test_settings_default_values():
    assert settings.PROJECT_NAME == "Argus ANPR Microservice"
    assert settings.VERSION  # non-empty — exact value varies with installed package
    assert settings.DEFAULT_PROVIDER.value in ["docling", "platerecognizer", "nvidia"]
    assert settings.HUMAN_CONF_THRESH == 0.30
    assert settings.VEHICLE_CONF_THRESH == 0.35


def test_anpr_service_error():
    err = ANPRServiceError("Base error message", status_code=500)
    assert err.message == "Base error message"
    assert err.status_code == 500
    assert str(err) == "Base error message"


def test_provider_not_found_error():
    err = ProviderNotFoundError("invalid_provider", ["docling", "nvidia"])
    assert err.status_code == 400
    assert "Unknown provider 'invalid_provider'" in err.message
    assert "docling, nvidia" in err.message


def test_invalid_image_error():
    err = InvalidImageError("Unsupported image format")
    assert err.status_code == 400
    assert err.message == "Unsupported image format"
