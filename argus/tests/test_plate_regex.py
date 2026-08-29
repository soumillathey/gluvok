"""
Regression tests for plate validation.

These exist because the pipeline previously used INDIAN_PLATE_REGEX.search(),
which matches a plate-shaped SUBSTRING anywhere in an OCR line. Text lifted off
a truck at a weighbridge routinely contains such substrings, so the service
returned brand names and chassis numbers as licence plates with a confident
SUCCESS status.

The false positives below are not hypothetical. "GOODYEAR2024" was reported as
plate "ODYEAR2024"; the stock-photo watermark in eval_report.json was reported
as "BP2A4904" with state "Bharat / Police / Custom Series".

If a future change reintroduces .search() anywhere in the validation path, these
tests fail.
"""

import pytest

from app.services.base import BasePlateRecognizer
from app.services.constants import INDIAN_PLATE_REGEX, STATE_CODES


class _Recognizer(BasePlateRecognizer):
    """Concrete subclass so parse_plate_info can be exercised directly."""

    def _recognize_single_image(self, image_input, filename="image.jpg"):
        return []


@pytest.fixture
def recognizer():
    return _Recognizer()


# Strings that plausibly appear on or near an Indian commercial vehicle.
# Every one of these was accepted as a licence plate before this fix.
NON_PLATE_STRINGS = [
    "GOODYEAR2024",  # tyre sidewall -> "ODYEAR2024"
    "ASHOKLEYLAND2820",  # cab badge     -> "LAND2820"
    "CHASSIS445566",  # stencilled ID -> "CHASSIS4455"
    "SRTRANSPORT2011",  # operator name -> "ANSPORT2011"
    "INSURANCEUPTO2026",  # sticker       -> "UPTO2026"
    "ALAMYDBP2A4904ALAMYIMAGEIDCRE1JFWWWALAMYCOM",  # raw unisolated watermark text
    "SPEED40KMLUCK",  # decal words
    "TATA2518DIESEL",  # model badge
    "OD00S5554",  # 00 is not valid district
    "MHABI1065",  # letter in district position
    "ARSS2557",  # letter in district position
]

# Strings that must still be accepted. Regressions here mean the fix was too strict.
REAL_PLATES = [
    "RJ09GA0165",
    "MH12AB1234",
    "RJ43GA2012",
    "NL02K7556",
    "OR02BU3389",
    "KA25B3155",
    "DL1CX2744",
    "GJ7UU1804",  # Single-digit district code
    "BP2A4904",  # BP series single-digit district
    "BP1A2453",  # BP series single-digit district
    "TN5ASS3555",  # Single-digit district code
    "22BH1234AA",  # Bharat series
]


@pytest.mark.parametrize("text", NON_PLATE_STRINGS)
def test_non_plate_text_is_rejected(recognizer, text):
    assert recognizer.parse_plate_info(text) is None, (
        f"{text!r} was accepted as a plate. A plate-shaped substring is not a plate."
    )


@pytest.mark.parametrize("plate", REAL_PLATES)
def test_real_plates_still_parse(recognizer, plate):
    info = recognizer.parse_plate_info(plate)
    assert info is not None, f"{plate!r} is a valid plate and must still parse"
    assert info["plate"] == plate
    assert info["state"] not in (None, "")


def test_bp_series_is_valid():
    """BP (Police / Government Series) is recognized with 1-digit district."""
    assert "BP" in STATE_CODES
    assert INDIAN_PLATE_REGEX.fullmatch("BP2A4904") is not None
    assert INDIAN_PLATE_REGEX.fullmatch("BP1A2453") is not None


def test_bh_series_is_retained():
    """BH (Bharat series) is real and has its own regex branch. Do not remove it."""
    assert "BH" in STATE_CODES
    assert INDIAN_PLATE_REGEX.fullmatch("22BH1234AA") is not None


def test_parse_returns_none_rather_than_unvalidated_text(recognizer):
    """
    parse_plate_info previously fell through to returning the raw OCR string as
    'plate' with a state guessed from its first two characters, so a function
    named 'parse_plate_info' had a path that validated nothing.
    """
    assert recognizer.parse_plate_info("HORNOKPLEASE") is None
    assert recognizer.parse_plate_info("") is None
    assert recognizer.parse_plate_info("XX") is None


def test_w8_to_wb_correction_survives(recognizer):
    """A known OCR confusion (W8 -> WB) is corrected before validation."""
    info = recognizer.parse_plate_info("W812AB1234")
    assert info is not None
    assert info["plate"] == "WB12AB1234"
    assert info["state"] == "West Bengal"


def test_validation_path_uses_fullmatch_not_search():
    """
    Guard against the specific regression this whole file exists to prevent.
    """
    import inspect

    from app.services import base
    from app.services.strategies import docling_ocr

    for module in (base, docling_ocr):
        source = inspect.getsource(module)
        assert "INDIAN_PLATE_REGEX.search(" not in source, (
            f"{module.__name__} uses INDIAN_PLATE_REGEX.search(). Use .fullmatch() — "
            "substring matching is how brand names become licence plates."
        )
