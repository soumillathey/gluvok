"""Dedicated coverage for DoclingStrategy's OCR-token parsing logic.

tests/test_services.py mocks the engine using RapidOCR's old tuple-response
shape. The engine actually installed (rapidocr's RapidOCROutput dataclass)
returns .txts/.scores/.boxes attributes instead, which is the branch these
tests exercise.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.services.image_processing import load_rgb
from app.services.strategies.docling_ocr import DoclingStrategy


def _mock_engine(txts, scores, boxes):
    engine = MagicMock()
    engine.return_value = SimpleNamespace(txts=txts, scores=scores, boxes=boxes)
    return engine


@patch("app.services.strategies.docling_ocr.get_docling_engine")
def test_single_token_exact_plate_match(mock_get_engine, sample_image_bytes):
    mock_get_engine.return_value = _mock_engine(
        txts=["RJ09GA0165"],
        scores=[0.95],
        boxes=[[0, 0, 100, 30]],
    )
    strategy = DoclingStrategy()
    result = strategy._extract_plates_from_image_array(load_rgb(sample_image_bytes))

    assert result[0]["plate"] == "RJ09GA0165"
    assert result[0]["state"] == "Rajasthan"


@patch("app.services.strategies.docling_ocr.get_docling_engine")
def test_two_line_spatial_pairing(mock_get_engine, sample_image_bytes):
    # Plate split across two OCR lines, e.g. state+district on one line,
    # series+serial directly below it.
    mock_get_engine.return_value = _mock_engine(
        txts=["RJ09", "GA0165"],
        scores=[0.90, 0.92],
        boxes=[[0, 0, 40, 20], [0, 22, 60, 42]],
    )
    strategy = DoclingStrategy()
    result = strategy._extract_plates_from_image_array(load_rgb(sample_image_bytes))

    assert result[0]["plate"] == "RJ09GA0165"


@patch("app.services.strategies.docling_ocr.get_docling_engine")
def test_decal_words_filtered_no_false_match(mock_get_engine, sample_image_bytes):
    mock_get_engine.return_value = _mock_engine(
        txts=["ASHOK LEYLAND", "TRANSPORT"],
        scores=[0.90, 0.90],
        boxes=[[0, 0, 100, 20], [0, 22, 100, 42]],
    )
    strategy = DoclingStrategy()
    result = strategy._extract_plates_from_image_array(load_rgb(sample_image_bytes))

    assert result[0]["plate"] == "N/A"


@patch("app.services.strategies.docling_ocr.get_docling_engine")
def test_no_ocr_tokens_returns_empty(mock_get_engine, sample_image_bytes):
    mock_get_engine.return_value = _mock_engine(txts=[], scores=[], boxes=[])
    strategy = DoclingStrategy()
    result = strategy._extract_plates_from_image_array(load_rgb(sample_image_bytes))

    assert result == []


@patch("app.services.strategies.docling_ocr.get_docling_engine")
def test_low_confidence_token_discarded(mock_get_engine, sample_image_bytes):
    mock_get_engine.return_value = _mock_engine(
        txts=["RJ09GA0165"],
        scores=[0.05],  # below the 0.20 score floor
        boxes=[[0, 0, 100, 30]],
    )
    strategy = DoclingStrategy()
    result = strategy._extract_plates_from_image_array(load_rgb(sample_image_bytes))

    assert result[0]["plate"] == "N/A"
