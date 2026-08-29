import os
from unittest.mock import patch

from app.schemas.plate import RecognitionResponse, RecognitionStatusEnum


def test_recognize_empty_file(client):
    response = client.post(
        "/recognize",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )
    assert response.status_code == 400
    data = response.json()
    assert data["success"] is False
    assert "empty" in data["message"].lower()


def test_recognize_invalid_image_bytes(client):
    response = client.post(
        "/recognize",
        files={"file": ("corrupt.jpg", b"not-a-valid-image-stream", "image/jpeg")},
    )
    assert response.status_code == 400
    data = response.json()
    assert data["success"] is False


def test_recognize_payload_too_large(client):
    # Simulate a file larger than MAX_UPLOAD_BYTES
    with patch("app.core.config.settings.MAX_UPLOAD_BYTES", 100):
        large_bytes = b"X" * 200
        response = client.post(
            "/recognize",
            files={"file": ("large.jpg", large_bytes, "image/jpeg")},
        )
        assert response.status_code == 413
        data = response.json()
        assert data["success"] is False
        assert "exceeds" in data["message"].lower()


def test_recognize_sample_image(client, sample_image_bytes):
    response = client.post(
        "/recognize",
        files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    validated = RecognitionResponse.model_validate(data)
    assert validated.filename == "test.jpg"
    # Plain red box has no 4-wheeler vehicle, default policy proceeds to OCR where no plate is found
    assert validated.vehicle_detected is False
    assert validated.status == RecognitionStatusEnum.NO_PLATE_DETECTED
    assert validated.rejected is False
    assert validated.success is False


def test_recognize_sample_image_rejected_when_policy_enabled(client, sample_image_bytes):
    with patch("app.services.yolo_filter.settings.REJECT_ON_NO_VEHICLE", True):
        response = client.post(
            "/recognize",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
        )
        assert response.status_code == 200
        data = response.json()
        validated = RecognitionResponse.model_validate(data)
        assert validated.filename == "test.jpg"
        assert validated.vehicle_detected is False
        assert validated.status == RecognitionStatusEnum.REJECTED_NO_FOUR_WHEELER
        assert validated.rejected is True


def test_recognize_real_image_if_present(client):
    image_path = os.path.join("tests", "1.jpg")
    if not os.path.exists(image_path):
        return

    with open(image_path, "rb") as f:
        img_bytes = f.read()

    response = client.post(
        "/recognize",
        files={"file": ("1.jpg", img_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    validated = RecognitionResponse.model_validate(data)
    assert validated.filename == "1.jpg"
    assert isinstance(validated.rejected, bool)
    assert validated.execution_time_ms is not None
