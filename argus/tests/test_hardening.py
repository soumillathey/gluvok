"""
Tests for runtime contracts, bounds, resource lifecycle, and error isolation.
"""

import io
import threading
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from app.core.contracts import ContractViolation, bounded, ensure, require
from app.schemas.plate import ProviderEnum
from app.services.pipeline import validate_plate_results


def _jpeg(width: int, height: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (100, 100, 100)).save(buf, format="JPEG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Runtime contracts (preconditions and postconditions)
# ---------------------------------------------------------------------------


def test_require_and_ensure_raise_on_violation():
    with pytest.raises(ContractViolation, match="precondition"):
        require(False, "box must be non-empty")
    with pytest.raises(ContractViolation, match="postcondition"):
        ensure(False, "crop must be non-empty")


def test_require_passes_on_truthy():
    require(1, "fine")
    ensure("yes", "fine")


def test_contracts_survive_optimised_mode():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-O",
            "-c",
            (
                "from app.core.contracts import require, ContractViolation\n"
                "try: require(False, 'test'); print('LIVED')\n"
                "except ContractViolation: print('RAISED')\n"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert "RAISED" in result.stdout, "contract check did not fire under -O; it has regressed to assert semantics"


# ---------------------------------------------------------------------------
# Fixed loop and sequence bounds
# ---------------------------------------------------------------------------


def test_bounded_truncates_and_preserves_order():
    assert bounded(list(range(100)), 3, "things") == [0, 1, 2]


def test_bounded_passes_short_sequences_through():
    assert bounded([1, 2], 5, "things") == [1, 2]
    assert bounded([], 5, "things") == []
    assert bounded(None, 5, "things") == []


def test_bounded_rejects_a_nonsense_limit():
    with pytest.raises(ContractViolation):
        bounded([1, 2, 3], 0, "things")


def test_vehicle_boxes_are_capped_in_the_waterfall():
    from app.core.config import settings
    from app.services.base import BasePlateRecognizer

    attempts = []

    class _Counting(BasePlateRecognizer):
        def _recognize_single_image(self, image_input, filename="image.jpg"):
            attempts.append(filename)
            return []

    many_boxes = [(0, 0, 100 - i, 100 - i) for i in range(20)]
    _Counting().recognize(_jpeg(400, 400), filename="x.jpg", vehicle_boxes=many_boxes)

    ceiling = settings.MAX_VEHICLE_BOXES * 5 * 2 + 2
    assert len(attempts) <= ceiling


def test_largest_boxes_are_the_ones_kept():
    from app.services.base import BasePlateRecognizer
    from app.services.image_processing import box_area

    seen_sizes = []

    class _Recorder(BasePlateRecognizer):
        def _recognize_single_image(self, image_input, filename="image.jpg"):
            with io.BytesIO(image_input) as buf, Image.open(buf) as img:
                seen_sizes.append(img.size)
            return []

    small = (0, 0, 20, 20)
    large = (0, 0, 300, 300)
    _Recorder().recognize(_jpeg(400, 400), vehicle_boxes=[small, large])

    assert box_area(large) > box_area(small)
    assert max(s[0] for s in seen_sizes) > 100


# ---------------------------------------------------------------------------
# Thread-safe model singleton initialization
# ---------------------------------------------------------------------------


def test_yolo_singleton_is_built_once_under_concurrency():
    import app.services.yolo_filter as yf

    builds = []

    def slow_build(_name):
        import time

        time.sleep(0.05)
        builds.append(1)
        return MagicMock()

    original = yf._YOLO_MODEL
    try:
        yf._YOLO_MODEL = None
        with patch.object(yf, "YOLO", side_effect=slow_build):
            threads = [threading.Thread(target=yf.get_yolo_model) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        assert len(builds) == 1, f"YOLO model built {len(builds)} times; the lock is not holding"
    finally:
        yf._YOLO_MODEL = original


# ---------------------------------------------------------------------------
# Parameter validation and boundary clamping
# ---------------------------------------------------------------------------


def test_out_of_bounds_box_is_clamped():
    from app.services.image_processing import clamp_box

    assert clamp_box((-50, -50, 5000, 5000), 640, 480) == (0, 0, 640, 480)


def test_corner_swapped_box_is_repaired():
    from app.services.image_processing import clamp_box

    assert clamp_box((300, 200, 100, 50), 640, 480) == (100, 50, 300, 200)


@pytest.mark.parametrize("bad", [None, (), (1, 2), (0, 0, 2, 2), "nope", (0, 0, "x", 4)])
def test_unusable_boxes_are_rejected_not_cropped(bad):
    from app.services.image_processing import clamp_box

    assert clamp_box(bad, 640, 480) is None


def test_clamp_rejects_nonsense_image_dimensions():
    from app.services.image_processing import clamp_box

    with pytest.raises(ContractViolation):
        clamp_box((0, 0, 10, 10), 0, 480)


def test_malformed_provider_output_does_not_error():
    mixed = [
        {"plate": "RJ09GA0165", "state": "Rajasthan"},
        {"unexpected_key": "boom"},
        "not a dict",
        {"plate": "MH12AB1234"},
    ]
    results = validate_plate_results(mixed, ProviderEnum.DOCLING)

    assert [r.plate for r in results] == ["RJ09GA0165", "MH12AB1234"]


def test_non_list_provider_output_is_handled():
    assert validate_plate_results({"plate": "X"}, ProviderEnum.NVIDIA) == []
    assert validate_plate_results(None, ProviderEnum.NVIDIA) == []


# ---------------------------------------------------------------------------
# Safe nested API response parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"choices": []},
        {"choices": [{}]},
        {"choices": [{"message": {}}]},
        {"choices": [{"message": {"content": None}}]},
        {"choices": [{"message": {"content": "   "}}]},
        {"error": {"message": "quota exceeded"}},
        {"choices": "not-a-list"},
        [],
        None,
    ],
)
def test_malformed_nvidia_response_returns_none_not_an_exception(payload):
    from app.services.strategies.nvidia_vision import extract_message_content

    assert extract_message_content(payload) is None


def test_wellformed_nvidia_response_is_extracted():
    from app.services.strategies.nvidia_vision import extract_message_content

    payload = {"choices": [{"message": {"content": "  RJ09GA0165 \n"}}]}
    assert extract_message_content(payload) == "RJ09GA0165"


# ---------------------------------------------------------------------------
# Bounded and promptly released resources
# ---------------------------------------------------------------------------


def test_image_loading_closes_its_source_handle():
    from app.services.image_processing import load_rgb

    buf = io.BytesIO(_jpeg(64, 64))
    img = load_rgb(buf.getvalue())

    assert img.size == (64, 64)
    assert img.mode == "RGB"
    assert getattr(img, "fp", None) is None


def test_no_bare_image_open_outside_the_helper():
    import inspect

    from app.services import image_processing
    from app.services.strategies import docling_ocr

    source = inspect.getsource(docling_ocr)
    assert "Image.open(" not in source

    ip_source = inspect.getsource(image_processing)
    for line in ip_source.splitlines():
        if "Image.open(" in line:
            assert "with " in line, f"unmanaged Image.open: {line.strip()}"
