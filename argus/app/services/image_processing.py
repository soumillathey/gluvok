import io

import cv2
import numpy as np
from PIL import Image, ImageFile, ImageOps

from app.core.config import settings
from app.core.contracts import ensure, require
from app.core.exceptions import InvalidImageError, PayloadTooLargeError

# Python 3.12 Type Aliases
type BoundingBox = tuple[int, int, int, int]
type ImageInput = str | bytes | Image.Image | np.ndarray

# Decompression-bomb guard. Pillow's default limit only emits a warning;
# this makes an oversized image raise before allocation.
Image.MAX_IMAGE_PIXELS = settings.MAX_IMAGE_PIXELS
object.__setattr__(ImageFile, "LOAD_TRUNCATED_IMAGES", True)

# A crop narrower than this cannot contain a readable plate. Used to reject
# degenerate boxes rather than feeding a 2-pixel sliver to downstream models.
MIN_CROP_EDGE_PX = 8


def decode_and_downscale(
    image_bytes: bytes,
    max_edge: int | None = None,
) -> bytes:
    """
    Validate an uploaded image and return normalised JPEG bytes.
    Guards decode against decompression bombs, applies EXIF orientation, and
    downscales so the longest edge is at most `max_edge`.
    """
    max_edge = max_edge or settings.MAX_IMAGE_EDGE_PX
    require(max_edge > 0, f"max_edge must be positive, got {max_edge}")
    require(bool(image_bytes), "decode_and_downscale received empty bytes")

    # Header probe without full pixel buffer allocation
    try:
        with io.BytesIO(image_bytes) as buf, Image.open(buf) as probe:
            width, height = probe.size
    except Image.DecompressionBombError as exc:
        raise PayloadTooLargeError(f"Image dimensions exceed permitted budget: {exc}") from exc
    except Exception as exc:
        raise InvalidImageError(f"Could not decode uploaded image: {exc}") from exc

    if width * height > settings.MAX_IMAGE_PIXELS:
        raise PayloadTooLargeError(
            f"Image is {width}x{height} ({width * height} pixels); limit is {settings.MAX_IMAGE_PIXELS} pixels."
        )

    try:
        pil_img = load_rgb(image_bytes)
    except Image.DecompressionBombError as exc:
        raise PayloadTooLargeError(f"Image dimensions exceed permitted budget: {exc}") from exc
    except Exception as exc:
        raise InvalidImageError(f"Could not decode uploaded image: {exc}") from exc

    if max(pil_img.size) > max_edge:
        pil_img.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)

    ensure(min(pil_img.size) > 0, "downscaled image collapsed to zero size")
    ensure(max(pil_img.size) <= max_edge, f"downscale failed to bound edge to {max_edge}")
    return _to_jpeg_bytes(pil_img)


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 contour points: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def warp_perspective_crop(img_bytes: bytes) -> bytes:
    """
    Detects perspective distortion / quad angle in an image crop and returns
    a perspective-warped, de-skewed frontal rectangular view as JPEG bytes.
    If no significant angle/distortion is detected, returns original JPEG bytes.
    """
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        return img_bytes

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Edge & Contour Detection for 4-point quadrilateral (plate or bumper bounds)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 50, 200)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # 4-point polygon quadrilateral with area > 5% of crop
        if len(approx) == 4 and cv2.contourArea(approx) > (0.05 * w * h):
            pts = approx.reshape(4, 2)
            rect = order_points(pts)
            tl, tr, br, bl = rect

            width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            max_width = max(int(width_a), int(width_b))

            height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            max_height = max(int(height_a), int(height_b))

            if max_width > 20 and max_height > 10:
                dst = np.array(
                    [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]], dtype="float32"
                )

                M = cv2.getPerspectiveTransform(rect, dst)
                warped = cv2.warpPerspective(img, M, (max_width, max_height))

                success, encoded = cv2.imencode(".jpg", warped)
                if success:
                    return bytes(encoded.tobytes())

    # Rotation De-skewing fallback using minAreaRect
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_c) > (0.05 * w * h):
            min_rect = cv2.minAreaRect(largest_c)
            angle = min_rect[-1]
            if angle < -45:
                angle = 90 + angle

            if abs(angle) > 3.0:
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

                success, encoded = cv2.imencode(".jpg", rotated)
                if success:
                    return bytes(encoded.tobytes())

    return img_bytes


def box_area(box: BoundingBox | None) -> int:
    """Area of an xyxy box, 0 for anything malformed."""
    if not box or len(box) != 4:
        return 0
    x1, y1, x2, y2 = box
    return max(0, int(x2) - int(x1)) * max(0, int(y2) - int(y1))


def clamp_box(
    box: BoundingBox | None,
    width: int,
    height: int,
) -> BoundingBox | None:
    """
    Clamp an xyxy box to real image bounds.
    Returns None when the box is malformed or degenerate after clamping.
    """
    require(width > 0 and height > 0, f"image dimensions must be positive, got {width}x{height}")

    if not box or len(box) != 4:
        return None

    try:
        x1, y1, x2, y2 = (int(v) for v in box)
    except (TypeError, ValueError):
        return None

    # Handle corner-swapped coordinates
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))

    if x2 - x1 < MIN_CROP_EDGE_PX or y2 - y1 < MIN_CROP_EDGE_PX:
        return None

    return (x1, y1, x2, y2)


def load_rgb(image_input: ImageInput) -> Image.Image:
    """
    Decode to an oriented RGB image, releasing the source handle immediately.
    """
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")

    if isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 2:
            return Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB))
        return Image.fromarray(
            cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB) if image_input.shape[2] == 3 else image_input
        )

    if isinstance(image_input, bytes):
        with io.BytesIO(image_input) as buf, Image.open(buf) as img:
            img.load()
            oriented = ImageOps.exif_transpose(img)
            return oriented.convert("RGB")

    with open(image_input, "rb") as fh, Image.open(fh) as img:
        img.load()
        oriented = ImageOps.exif_transpose(img)
        return oriented.convert("RGB")


def _to_jpeg_bytes(img: Image.Image, quality: int = 90) -> bytes:
    with io.BytesIO() as buf:
        img.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()


def crop_image_roi(
    image_input: ImageInput,
    vehicle_box: BoundingBox | None = None,
    bottom_crop_ratio: float = 0.50,
    bottom_roi_only: bool = True,
) -> bytes:
    """
    Extract cropped image bytes from an image input.
    - If vehicle_box is provided: crops the vehicle bounding box, clamped to bounds.
    - If bottom_roi_only is True: keeps bottom `bottom_crop_ratio` of that crop.
    """
    require(
        0.0 < bottom_crop_ratio <= 1.0,
        f"bottom_crop_ratio must be in (0, 1], got {bottom_crop_ratio}",
    )

    pil_img = load_rgb(image_input)
    w, h = pil_img.size

    crop_box = clamp_box(vehicle_box, w, h)
    if crop_box is not None:
        cropped = pil_img.crop(crop_box)
        if bottom_roi_only:
            vw, vh = cropped.size
            v_crop_top = int(vh * (1.0 - bottom_crop_ratio))
            cropped = cropped.crop((0, v_crop_top, vw, vh))
        ensure(min(cropped.size) > 0, "vehicle ROI crop collapsed to zero size")
        return _to_jpeg_bytes(cropped)

    # No box: bottom band of whole frame
    crop_top = int(h * (1.0 - bottom_crop_ratio)) if bottom_roi_only else 0
    cropped = pil_img.crop((0, crop_top, w, h))
    ensure(min(cropped.size) > 0, "full-frame ROI crop collapsed to zero size")
    return _to_jpeg_bytes(cropped)
