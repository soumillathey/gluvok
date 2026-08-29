import os
from fastapi.testclient import TestClient
from app.server import app

client = TestClient(app)


def test_hermes_to_argus_contract():
    """
    Validates that the payload Hermes sends (multipart 'file' with image/jpeg)
    is accepted by Argus's FastAPI endpoint and yields a valid response.
    """
    image_path = os.path.join("tests", "1.jpg")
    if not os.path.exists(image_path):
        return

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    # Exact payload structure sent by hermes-camera3 src/camera/anpr_client.py
    files = {"file": ("frame.jpg", image_bytes, "image/jpeg")}
    response = client.post("/recognize", files=files)

    assert response.status_code == 200
    data = response.json()

    assert "results" in data
    assert "success" in data
    assert "status" in data
    assert "execution_time_ms" in data

    # Verify that the schema matches what Hermes anpr_client.py expects to extract
    if data["success"] and data["results"]:
        plate_str = data["results"][0]["plate"]
        assert isinstance(plate_str, str)
        assert len(plate_str) > 0


def test_hermes_legacy_image_field_contract():
    """
    Validates that if a client sends 'image' instead of 'file', Argus also handles it cleanly.
    """
    image_path = os.path.join("tests", "1.jpg")
    if not os.path.exists(image_path):
        return

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    files = {"image": ("frame.jpg", image_bytes, "image/jpeg")}
    response = client.post("/recognize", files=files)

    assert response.status_code == 200
    data = response.json()
    assert "status" in data
