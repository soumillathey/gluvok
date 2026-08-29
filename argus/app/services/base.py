import io
import re
from abc import ABC, abstractmethod
from typing import Any

from app.core.config import settings
from app.core.contracts import bounded, require
from app.services.constants import INDIAN_PLATE_REGEX, STATE_CODES
from app.services.image_processing import (
    BoundingBox,
    ImageInput,
    box_area,
    crop_image_roi,
    load_rgb,
    warp_perspective_crop,
)


class BasePlateRecognizer(ABC):
    """
    Abstract Base Class for ANPR Recognition Strategies.
    Provides standard vehicle crop hierarchy, de-skew fallback, and state resolution.
    """

    @abstractmethod
    def _recognize_single_image(self, image_input: str | bytes, filename: str = "image.jpg") -> list[dict[str, Any]]:
        """
        Subclasses implement OCR / VLM plate extraction on a single image buffer.
        Returns a list of dicts with keys: 'plate', 'state', 'raw_text'.
        """

    def recognize(
        self,
        image_input: ImageInput,
        filename: str = "image.jpg",
        vehicle_box: BoundingBox | None = None,
        vehicle_boxes: list[BoundingBox] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Hierarchical ANPR recognition:
          1. Vehicle Bounding Box Crop (with perspective de-skew fallback)
          2. Lower 60% Bumper Crop (isolates plate from top cab text/decals)
          3. Full Frame Fallback
        """
        require(image_input is not None, "recognize() called with no image")

        image_bytes: bytes
        if isinstance(image_input, str):
            filename = filename or image_input
            with open(image_input, "rb") as f:
                image_bytes = f.read()
        elif isinstance(image_input, bytes):
            image_bytes = image_input
        else:
            # Image.Image / ndarray input: encode to JPEG bytes so downstream
            # warp_perspective_crop (which needs a real byte buffer) works.
            with io.BytesIO() as buf:
                load_rgb(image_input).save(buf, format="JPEG")
                image_bytes = buf.getvalue()

        raw_text_fallbacks: list[list[dict[str, Any]]] = []

        # Determine candidate vehicle bounding boxes (largest first)
        candidate_boxes = vehicle_boxes if vehicle_boxes else ([vehicle_box] if vehicle_box else [])
        ordered = sorted((b for b in candidate_boxes if b), key=box_area, reverse=True)
        boxes_to_check: list[BoundingBox | None] = list(bounded(ordered, settings.MAX_VEHICLE_BOXES, "vehicle boxes"))
        if not boxes_to_check:
            boxes_to_check = [None]

        def _try_crop(crop_bytes: bytes) -> list[dict[str, Any]] | None:
            res = self._recognize_single_image(crop_bytes, filename=filename)
            if any(r.get("plate") and r.get("plate") != "N/A" for r in res):
                return res
            if res:
                raw_text_fallbacks.append(res)

            # Try perspective de-skewed crop if significant angle detected
            warped_bytes = warp_perspective_crop(crop_bytes)
            if warped_bytes != crop_bytes:
                res_warped = self._recognize_single_image(warped_bytes, filename=filename)
                if any(r.get("plate") and r.get("plate") != "N/A" for r in res_warped):
                    return res_warped
                if res_warped:
                    raw_text_fallbacks.append(res_warped)
            return None

        detected_plates: list[dict[str, Any]] = []
        seen_plate_nums = set()

        # Step 1 & 2: Vehicle Crop & Lower Bumper Crop across each detected vehicle
        for box in boxes_to_check:
            if box:
                # 1. Full vehicle crop
                veh_crop = crop_image_roi(image_bytes, box, bottom_crop_ratio=1.0, bottom_roi_only=False)
                res = _try_crop(veh_crop)
                if res:
                    for r in res:
                        p = r.get("plate")
                        if p and p != "N/A" and p not in seen_plate_nums:
                            seen_plate_nums.add(p)
                            detected_plates.append(r)
                    continue

                # 2. Lower bumper crop (50% bottom)
                bumper_crop = crop_image_roi(image_bytes, box, bottom_crop_ratio=0.50, bottom_roi_only=True)
                res_bumper = _try_crop(bumper_crop)
                if res_bumper:
                    for r in res_bumper:
                        p = r.get("plate")
                        if p and p != "N/A" and p not in seen_plate_nums:
                            seen_plate_nums.add(p)
                            detected_plates.append(r)

        if detected_plates:
            return detected_plates

        # Step 3: Full original frame fallback
        res_full = _try_crop(image_bytes)
        if res_full:
            return res_full

        return raw_text_fallbacks[0] if raw_text_fallbacks else []

    def parse_plate_info(self, raw_plate: str | None) -> dict[str, Any] | None:
        """
        Validate candidate plate string against Indian plate regex and resolve State/UT.
        Strips whitespace and normalizes known state prefix confusions (e.g. W8 -> WB).
        """
        if not raw_plate:
            return None

        cleaned = re.sub(r"[^A-Za-z0-9]", "", str(raw_plate)).upper()
        if not cleaned:
            return None

        if cleaned.startswith("W8"):
            cleaned = "WB" + cleaned[2:]

        match = INDIAN_PLATE_REGEX.fullmatch(cleaned)
        if not match:
            return None

        matched_plate = cleaned

        # Standard Indian Series State resolution
        if match.group(1):
            state_code = match.group(1).upper()
            state_name = STATE_CODES.get(state_code, "Unknown State")
            return {"plate": matched_plate, "state": state_name}

        # Bharat (BH) Series
        if match.group(5):  # "BH"
            return {"plate": matched_plate, "state": STATE_CODES.get("BH", "Bharat Series (National)")}

        return {"plate": matched_plate, "state": "Unknown State"}
