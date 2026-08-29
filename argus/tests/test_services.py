from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from app.core.exceptions import ProviderNotFoundError
from app.schemas.plate import (
    BoundingBox,
    ProviderEnum,
    RecognitionStatusEnum,
)
from app.services.base import BasePlateRecognizer
from app.services.constants import INDIAN_PLATE_REGEX, STATE_CODES
from app.services.factory import PlateRecognizerFactory
from app.services.image_processing import ImageInput
from app.services.strategies.docling_ocr import (
    DoclingStrategy,
    normalize_candidate_strings,
)
from app.services.strategies.nvidia_vision import NvidiaVisionStrategy
from app.services.strategies.plate_recognizer import PlateRecognizerStrategy
from app.services.yolo_filter import filter_vehicle_and_occupancy


class DummyStrategy(BasePlateRecognizer):
    """Minimal concrete subclass for factory and parse tests."""

    def _recognize_single_image(
        self,
        image_input: str | bytes,
        filename: str = "image.jpg",
    ) -> list[dict[str, Any]]:
        return [{"plate": "MH12AB1234", "state": "Maharashtra"}]

    def recognize(
        self,
        image_input: ImageInput,
        filename: str = "image.jpg",
        vehicle_box: BoundingBox | None = None,
        vehicle_boxes: list[BoundingBox] | None = None,
    ) -> list[dict[str, Any]]:
        return [{"plate": "MH12AB1234", "state": "Maharashtra"}]


def test_indian_plate_regex_and_state_codes():
    # Valid Indian plate patterns
    plates_to_test = [
        ("RJ09GA0165", "Rajasthan"),
        ("MH12AB1234", "Maharashtra"),
        ("DL01C1234", "Delhi"),
        ("KA05MB9999", "Karnataka"),
        ("TN07AZ0001", "Tamil Nadu"),
    ]
    for plate_str, expected_state in plates_to_test:
        match = INDIAN_PLATE_REGEX.search(plate_str)
        assert match is not None
        state_code = match.group(1) or match.group(6)
        assert STATE_CODES.get(state_code) == expected_state


def test_base_plate_recognizer_parse_plate_info():
    strategy = DummyStrategy()

    # Test valid regex match
    info = strategy.parse_plate_info("  rj 09 ga 0165 ")
    assert info is not None
    assert info["plate"] == "RJ09GA0165"
    assert info["state"] == "Rajasthan"

    # Test none / empty input
    assert strategy.parse_plate_info("") is None
    assert strategy.parse_plate_info(None) is None

    # Test invalid plate input returns None (no unvalidated fallback)
    assert strategy.parse_plate_info("XX999999") is None


def test_factory_list_and_get():
    providers = PlateRecognizerFactory.list_providers()
    assert ProviderEnum.DOCLING in providers
    assert ProviderEnum.NVIDIA in providers
    assert ProviderEnum.PLATERECOGNIZER in providers

    # Valid get
    recognizer = PlateRecognizerFactory.get_recognizer(ProviderEnum.DOCLING)
    assert isinstance(recognizer, DoclingStrategy)

    # String input get
    recognizer_str = PlateRecognizerFactory.get_recognizer("nvidia")
    assert isinstance(recognizer_str, NvidiaVisionStrategy)

    # Unknown provider string throws ProviderNotFoundError
    with pytest.raises(ProviderNotFoundError):
        PlateRecognizerFactory.get_recognizer("unknown_provider")


@patch("app.services.yolo_filter.get_yolo_model")
def test_yolo_filter_eligible_vehicle(mock_get_model, sample_image_bytes):
    mock_box_car = MagicMock()
    mock_box_car.__len__.return_value = 1
    mock_box_car.cls.cpu().numpy.return_value = [2]  # Class 2 = car
    mock_box_car.conf.cpu().numpy.return_value = [0.90]
    mock_box_car.xyxy.cpu().numpy.return_value = [[10, 10, 90, 90]]

    mock_results = MagicMock()
    mock_results.boxes = mock_box_car

    mock_model = MagicMock()
    mock_model.return_value = [mock_results]
    mock_get_model.return_value = mock_model

    res = filter_vehicle_and_occupancy(sample_image_bytes)
    assert res["is_eligible"] is True
    assert res["vehicle_detected"] is True
    assert res["vehicle_type"] == "car"
    assert res["human_detected"] is False


@patch("app.services.yolo_filter.get_yolo_model")
def test_yolo_filter_human_detection_policy(mock_get_model, sample_image_bytes):
    mock_box_human = MagicMock()
    mock_box_human.__len__.return_value = 2
    mock_box_human.cls.cpu().numpy.return_value = [0, 2]  # Class 0 = person, 2 = car
    mock_box_human.conf.cpu().numpy.return_value = [0.85, 0.90]
    mock_box_human.xyxy.cpu().numpy.return_value = [[5, 5, 15, 15], [10, 10, 90, 90]]

    mock_results = MagicMock()
    mock_results.boxes = mock_box_human

    mock_model = MagicMock()
    mock_model.return_value = [mock_results]
    mock_get_model.return_value = mock_model

    # Default policy: reject_on_human is False -> eligible
    res_default = filter_vehicle_and_occupancy(sample_image_bytes)
    assert res_default["is_eligible"] is True
    assert res_default["status"] is None
    assert res_default["human_detected"] is True
    assert res_default["vehicle_detected"] is True

    # Explicit policy: reject_on_human is True -> rejected
    res_rejected = filter_vehicle_and_occupancy(sample_image_bytes, reject_on_human=True)
    assert res_rejected["is_eligible"] is False
    assert res_rejected["status"] == RecognitionStatusEnum.REJECTED_HUMAN_DETECTED
    assert res_rejected["human_detected"] is True


@patch("app.services.yolo_filter.get_yolo_model")
def test_yolo_filter_no_vehicle_policy(mock_get_model, sample_image_bytes):
    mock_box_empty = MagicMock()
    mock_box_empty.__len__.return_value = 0
    mock_box_empty.cls.cpu().numpy.return_value = []
    mock_box_empty.conf.cpu().numpy.return_value = []
    mock_box_empty.xyxy.cpu().numpy.return_value = []

    mock_results = MagicMock()
    mock_results.boxes = mock_box_empty

    mock_model = MagicMock()
    mock_model.return_value = [mock_results]
    mock_get_model.return_value = mock_model

    # Default policy: reject_on_no_vehicle is False -> eligible for direct plate OCR
    res_default = filter_vehicle_and_occupancy(sample_image_bytes)
    assert res_default["is_eligible"] is True
    assert res_default["status"] is None
    assert res_default["vehicle_detected"] is False
    assert res_default["vehicle_count"] == 0

    # Explicit policy: reject_on_no_vehicle is True -> rejected
    res_rejected = filter_vehicle_and_occupancy(sample_image_bytes, reject_on_no_vehicle=True)
    assert res_rejected["is_eligible"] is False
    assert res_rejected["status"] == RecognitionStatusEnum.REJECTED_NO_FOUR_WHEELER
    assert res_rejected["vehicle_detected"] is False
    assert res_rejected["vehicle_count"] == 0


@patch("app.services.yolo_filter.get_yolo_model")
def test_yolo_filter_multiple_vehicles_policy(mock_get_model, sample_image_bytes):
    mock_box_multiple = MagicMock()
    mock_box_multiple.__len__.return_value = 2
    mock_box_multiple.cls.cpu().numpy.return_value = [2, 7]  # Class 2 = car, 7 = truck
    mock_box_multiple.conf.cpu().numpy.return_value = [0.85, 0.90]
    mock_box_multiple.xyxy.cpu().numpy.return_value = [[10, 10, 50, 50], [50, 50, 90, 90]]

    mock_results = MagicMock()
    mock_results.boxes = mock_box_multiple

    mock_model = MagicMock()
    mock_model.return_value = [mock_results]
    mock_get_model.return_value = mock_model

    # Default policy: reject_on_multiple_vehicles is False -> eligible with primary vehicle
    res_default = filter_vehicle_and_occupancy(sample_image_bytes)
    assert res_default["is_eligible"] is True
    assert res_default["status"] is None
    assert res_default["vehicle_detected"] is True
    assert res_default["vehicle_count"] == 2

    # Explicit policy: reject_on_multiple_vehicles is True -> rejected
    res_rejected = filter_vehicle_and_occupancy(sample_image_bytes, reject_on_multiple_vehicles=True)
    assert res_rejected["is_eligible"] is False
    assert res_rejected["status"] == RecognitionStatusEnum.REJECTED_MULTIPLE_VEHICLES
    assert res_rejected["vehicle_detected"] is True
    assert res_rejected["vehicle_count"] == 2


@patch("app.services.strategies.docling_ocr.get_docling_engine")
def test_docling_ocr_strategy_mocked(mock_get_engine, sample_image_bytes):
    mock_engine = MagicMock()
    mock_engine.return_value = ([([0, 0, 100, 30], "RJ09GA0165", 0.98)], [0.01, 0.01, 0.01])
    mock_get_engine.return_value = mock_engine

    strategy = DoclingStrategy()
    results = strategy.recognize(sample_image_bytes)
    assert len(results) == 1
    assert results[0]["plate"] == "RJ09GA0165"
    assert results[0]["state"] == "Rajasthan"


@patch("requests.post")
def test_nvidia_vision_strategy_mocked(mock_post, sample_image_bytes):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"choices": [{"message": {"content": "The vehicle license plate is RJ09GA0165."}}]}
    mock_post.return_value = mock_resp

    strategy = NvidiaVisionStrategy(api_key="test_key")
    results = strategy.recognize(sample_image_bytes)
    assert len(results) == 1
    assert results[0]["plate"] == "RJ09GA0165"
    assert results[0]["state"] == "Rajasthan"


@patch("requests.post")
def test_plate_recognizer_strategy_mocked(mock_post, sample_image_bytes):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"results": [{"plate": "rj09ga0165", "candidates": [{"plate": "rj09ga0165"}]}]}
    mock_post.return_value = mock_resp

    strategy = PlateRecognizerStrategy(token="test_token")
    results = strategy.recognize(sample_image_bytes)
    assert len(results) == 1
    assert results[0]["plate"] == "RJ09GA0165"
    assert results[0]["state"] == "Rajasthan"


def test_plate_recognizer_skips_large_file():
    recognizer = PlateRecognizerStrategy(token="dummy_token")
    large_bytes = b"0" * int(3.6 * 1024 * 1024)
    res = recognizer._recognize_single_image(large_bytes)
    assert res == []


def test_normalize_candidate_strings():
    # State prefix corrections
    assert "WB12AB1234" in normalize_candidate_strings("W812AB1234")
    assert "RJ14GJ4976" in normalize_candidate_strings("RT14G34976")
    assert "RJ09GA0165" in normalize_candidate_strings("RJ09GA0165")

    # Positional character confusions (O/0, I/1, G3/GJ)
    variants = normalize_candidate_strings("RT14G34976")
    assert "RJ14GJ4976" in variants
