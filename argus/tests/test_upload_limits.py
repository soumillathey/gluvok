"""
Tests for payload budgets, downscaling, pixel caps, and HTTP timeouts.
"""

import inspect
import io

import pytest
from PIL import Image

from app.core.config import settings
from app.core.exceptions import InvalidImageError, PayloadTooLargeError
from app.services.image_processing import decode_and_downscale
from app.services.pipeline import recognize_plate_image


def _jpeg(width: int, height: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (120, 120, 120)).save(buf, format="JPEG")
    return buf.getvalue()


# --------------------------------------------------------------------------
# Upload size & limits
# --------------------------------------------------------------------------


def test_oversized_upload_is_rejected():
    oversized = b"\xff\xd8\xff\xe0" + b"\x00" * (settings.MAX_UPLOAD_BYTES + 1024)
    with pytest.raises(PayloadTooLargeError):
        recognize_plate_image(oversized, filename="huge.jpg")


def test_empty_upload_is_rejected():
    with pytest.raises(InvalidImageError):
        recognize_plate_image(b"", filename="empty.jpg")


# --------------------------------------------------------------------------
# Decode bombs and downscaling
# --------------------------------------------------------------------------


def test_pixel_budget_is_enforced(monkeypatch):
    """
    A small file can declare enormous dimensions. Guard on the pixel count from
    the header, before the full decode allocates anything.
    """
    monkeypatch.setattr(settings, "MAX_IMAGE_PIXELS", 1000)
    with pytest.raises(PayloadTooLargeError):
        decode_and_downscale(_jpeg(200, 200))


def test_large_image_is_downscaled():
    out = decode_and_downscale(_jpeg(4000, 3000))
    assert max(Image.open(io.BytesIO(out)).size) <= settings.MAX_IMAGE_EDGE_PX


def test_downscaled_output_fits_plate_recognizer_ceiling():
    assert len(decode_and_downscale(_jpeg(4000, 3000))) < 3.5 * 1024 * 1024


def test_small_image_is_not_upscaled():
    out = decode_and_downscale(_jpeg(320, 240))
    assert Image.open(io.BytesIO(out)).size == (320, 240)


def test_undecodable_bytes_raise_invalid_image():
    with pytest.raises(InvalidImageError):
        decode_and_downscale(b"this is not an image")


# --------------------------------------------------------------------------
# Outbound HTTP timeouts
# --------------------------------------------------------------------------


@pytest.mark.parametrize("module_name", ["nvidia_vision", "plate_recognizer"])
def test_provider_requests_specify_a_timeout(module_name):
    module = __import__(f"app.services.strategies.{module_name}", fromlist=[module_name])
    source = inspect.getsource(module)
    assert "requests.post(" in source
    assert "timeout=" in source, f"{module_name} calls requests.post without a timeout."


def test_timeout_settings_are_bounded():
    assert 0 < settings.HTTP_CONNECT_TIMEOUT <= 10
    assert 0 < settings.HTTP_READ_TIMEOUT <= 30
