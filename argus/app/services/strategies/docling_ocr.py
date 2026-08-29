import logging
import math
import re
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image
from rapidocr import RapidOCR

from app.core.config import settings
from app.core.contracts import bounded, require
from app.core.logging import logger
from app.services.base import BasePlateRecognizer
from app.services.constants import (
    INDIAN_PLATE_REGEX,
    NON_PLATE_WORDS,
    normalize_candidate_strings,
)
from app.services.image_processing import ImageInput, load_rgb

# Direct RapidOCR engine instance
_DOCLING_ENGINE = RapidOCR()

# Suppress RapidOCR internal per-crop empty detection warnings on logger and its handlers
_rapid_logger = logging.getLogger("RapidOCR")
_rapid_logger.setLevel(logging.ERROR)
for _h in _rapid_logger.handlers:
    _h.setLevel(logging.ERROR)


def get_docling_engine() -> RapidOCR:
    """Return the RapidOCR engine instance."""
    return _DOCLING_ENGINE


def check_docling_engine() -> bool:
    """Verify that the RapidOCR engine is operational."""
    return _DOCLING_ENGINE is not None


@dataclass(slots=True, frozen=True)
class OCRToken:
    """High-performance slotted container for extracted OCR tokens."""

    text: str
    score: float
    cx: float | None = None
    cy: float | None = None


@dataclass(slots=True)
class PlateCandidate:
    """Slotted candidate plate match with vertical spatial priority ranking."""

    y_pos: float
    rank: int
    info: dict[str, Any]


def _get_box_centroid(box: Any) -> tuple[float | None, float | None]:
    """Calculate (x_center, y_center) centroid of an OCR bounding box."""
    if box is None:
        return None, None
    try:
        if isinstance(box, (list, tuple, np.ndarray)) and len(box) >= 4:
            if all(isinstance(v, (int, float, np.number)) for v in box[:4]):
                return float((box[0] + box[2]) / 2.0), float((box[1] + box[3]) / 2.0)
            if all(isinstance(pt, (list, tuple, np.ndarray)) and len(pt) >= 2 for pt in box):
                pts_x = [pt[0] for pt in box]
                pts_y = [pt[1] for pt in box]
                return float(np.mean(pts_x)), float(np.mean(pts_y))
    except (TypeError, ValueError, IndexError, AttributeError):
        pass
    return None, None


def _is_decal_word(word: str) -> bool:
    """Check if a candidate string is a common commercial vehicle decal word."""
    return word in NON_PLATE_WORDS or any(w in word for w in ("CARRIER", "LEYLAND", "TRANSPORT", "NATIONALPERMIT"))


def _enhance_contrast(img: Image.Image) -> Image.Image:
    """Enhance local image contrast and resolution using CLAHE and upscaling for low-contrast/distant plates."""
    np_img = np.array(img)
    h, w = np_img.shape[:2]

    # If crop is low-resolution (e.g. distant truck bumper), upscale 2.5x with bicubic interpolation
    if w < 600 or h < 600:
        np_img = cv2.resize(np_img, (0, 0), fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)

    if len(np_img.shape) == 3:
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    else:
        gray = np_img

    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(4, 4))
    enhanced_gray = clahe.apply(gray)
    enhanced_rgb = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(enhanced_rgb)


class DoclingStrategy(BasePlateRecognizer):
    """
    Concrete ANPR Strategy using RapidOCR ONNX Runtime engine
    with 2D spatial layout parsing, slotted dataclass token evaluation, and vertical bumper candidate ranking.
    """

    def _extract_plates_from_image_array(self, img_pil: Image.Image) -> list[dict[str, Any]]:
        require(img_pil is not None, "_extract_plates_from_image_array received None")

        engine = get_docling_engine()
        np_img = np.array(img_pil)

        raw_items: list[OCRToken] = []

        try:
            res = engine(np_img)
            if not res:
                return []

            # RapidOCR v3.9+ returns RapidOCROutput with .txts, .scores, .boxes
            txts = getattr(res, "txts", None)
            scores = getattr(res, "scores", None)
            if txts and scores:
                boxes = getattr(res, "boxes", None)
                for idx, (t, s) in enumerate(zip(txts, scores, strict=False)):
                    cx, cy = _get_box_centroid(boxes[idx]) if (boxes is not None and idx < len(boxes)) else (None, None)
                    raw_items.append(OCRToken(text=str(t), score=float(s), cx=cx, cy=cy))

            elif isinstance(res, (tuple, list)) and len(res) == 2 and isinstance(res[0], (list, tuple)):
                for item in res[0]:
                    if len(item) >= 3:
                        box, text, score = item[0], item[1], item[2]
                        cx, cy = _get_box_centroid(box)
                        raw_items.append(OCRToken(text=str(text), score=float(score), cx=cx, cy=cy))

            elif isinstance(res, (tuple, list)):
                for item in res:
                    if isinstance(item, (list, tuple)) and len(item) >= 3:
                        box, text, score = item[0], item[1], item[2]
                        cx, cy = _get_box_centroid(box)
                        raw_items.append(OCRToken(text=str(text), score=float(score), cx=cx, cy=cy))
        except (RuntimeError, ValueError, TypeError, IndexError, AttributeError, OSError) as e:
            logger.error(f"[docling/rapidocr] OCR execution failed: {e}")
            return []

        if not raw_items:
            return []

        # Sort items spatially top-to-bottom if coordinates are available
        if any(item.cy is not None for item in raw_items):
            raw_items.sort(key=lambda x: x.cy if x.cy is not None else 9999.0)

        # Bounded OCR lines
        lines_data = list(bounded(raw_items, settings.MAX_OCR_LINES, "OCR text lines"))

        clean_tokens: list[OCRToken] = []
        raw_text_parts: list[str] = []
        seen_tokens = set()

        for token in lines_data:
            if not token.text or token.score < 0.20:
                continue
            raw_text_parts.append(token.text.strip())

            # Full line cleaned
            cleaned = re.sub(r"[^A-Za-z0-9]", "", token.text).upper()
            if cleaned and len(cleaned) >= 2 and not _is_decal_word(cleaned) and cleaned not in seen_tokens:
                seen_tokens.add(cleaned)
                clean_tokens.append(OCRToken(text=cleaned, score=token.score, cx=token.cx, cy=token.cy))

            # Sub-tokens (when a line contains multiple space-separated words)
            for part in token.text.split():
                part_cleaned = re.sub(r"[^A-Za-z0-9]", "", part).upper()
                if (
                    part_cleaned
                    and len(part_cleaned) >= 2
                    and not _is_decal_word(part_cleaned)
                    and part_cleaned not in seen_tokens
                ):
                    seen_tokens.add(part_cleaned)
                    clean_tokens.append(OCRToken(text=part_cleaned, score=token.score, cx=token.cx, cy=token.cy))

        raw_text_summary = " ".join(raw_text_parts) if raw_text_parts else "N/A"

        # Candidate collection with vertical position weighting
        plate_candidates: list[PlateCandidate] = []
        seen_matched_plates = set()

        # 1. Collect individual recognized text tokens
        for tok in clean_tokens:
            y_pos = tok.cy if tok.cy is not None else 0.0
            for rank, cand_norm in enumerate(normalize_candidate_strings(tok.text)):
                match = INDIAN_PLATE_REGEX.fullmatch(cand_norm)
                if match:
                    info = self.parse_plate_info(match.group(0))
                    if info:
                        plate_num = info.get("plate")
                        if plate_num and plate_num not in seen_matched_plates:
                            seen_matched_plates.add(plate_num)
                            info["raw_text"] = raw_text_summary
                            plate_candidates.append(PlateCandidate(y_pos=y_pos, rank=rank, info=info))

        # 2. Collect 2-line combinations sorted by 2D spatial proximity
        candidate_pairs: list[tuple[float, str, float]] = []  # (dist, text, y_pos)
        n = len(clean_tokens)
        for i in range(n):
            tok_a = clean_tokens[i]
            for j in range(i + 1, min(i + 6, n)):
                tok_b = clean_tokens[j]
                if tok_a.cx is not None and tok_a.cy is not None and tok_b.cx is not None and tok_b.cy is not None:
                    dist = math.hypot(tok_a.cx - tok_b.cx, tok_a.cy - tok_b.cy)
                    y_mean = float((tok_a.cy + tok_b.cy) / 2.0)
                else:
                    dist = float(abs(i - j) * 100.0)
                    y_mean = tok_a.cy or tok_b.cy or 0.0

                candidate_pairs.append((dist, tok_a.text + tok_b.text, y_mean))
                candidate_pairs.append((dist + 0.1, tok_b.text + tok_a.text, y_mean))

        # Sort candidate pairs by spatial proximity (closest first)
        candidate_pairs.sort(key=lambda p: p[0])

        for _, pair_raw, y_pos in candidate_pairs:
            for rank, pair_norm in enumerate(normalize_candidate_strings(pair_raw)):
                match = INDIAN_PLATE_REGEX.fullmatch(pair_norm)
                if match:
                    info = self.parse_plate_info(match.group(0))
                    if info:
                        plate_num = info.get("plate")
                        if plate_num and plate_num not in seen_matched_plates:
                            seen_matched_plates.add(plate_num)
                            info["raw_text"] = raw_text_summary
                            plate_candidates.append(PlateCandidate(y_pos=y_pos, rank=rank, info=info))

        # 3. Select best candidate: prioritize exact matches (rank 0) first,
        # then rank by lower-bumper vertical position descending (highest y)
        if plate_candidates:
            plate_candidates.sort(key=lambda c: (-c.rank, c.y_pos), reverse=True)
            return [plate_candidates[0].info]

        # 4. Fallback: If no valid plate matched
        return [{"plate": "N/A", "state": "N/A", "raw_text": raw_text_summary}]

    def _recognize_single_image(self, image_input: ImageInput, filename: str = "image.jpg") -> list[dict[str, Any]]:
        """Process an image input with Docling RapidOCR engine, with CLAHE enhancement fallback."""
        pil_img = load_rgb(image_input)
        res = self._extract_plates_from_image_array(pil_img)
        if any(r.get("plate") and r.get("plate") != "N/A" for r in res):
            return res

        # CLAHE Contrast Enhancement Fallback
        try:
            enhanced_img = _enhance_contrast(pil_img)
            res_enh = self._extract_plates_from_image_array(enhanced_img)
            if any(r.get("plate") and r.get("plate") != "N/A" for r in res_enh):
                return res_enh
            if res_enh:
                return res_enh
        except (cv2.error, ValueError, RuntimeError, OSError, TypeError) as e:
            logger.debug(f"Contrast enhancement fallback skipped: {e}")

        return res
