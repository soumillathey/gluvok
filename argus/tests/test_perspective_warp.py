import io

import cv2
import numpy as np
from PIL import Image

from app.services.image_processing import order_points, warp_perspective_crop


def test_order_points():
    pts = np.array([[100, 200], [10, 10], [200, 10], [10, 200]], dtype="float32")

    rect = order_points(pts)
    assert np.allclose(rect[0], [10, 10])  # Top-Left
    assert np.allclose(rect[1], [200, 10])  # Top-Right
    assert np.allclose(rect[2], [100, 200])  # Bottom-Right
    assert np.allclose(rect[3], [10, 200])  # Bottom-Left


def test_warp_perspective_crop_returns_valid_jpeg():
    # Create a synthetic image with a skewed rectangle
    img = np.ones((300, 400, 3), dtype=np.uint8) * 255
    pts = np.array([[50, 80], [350, 50], [320, 250], [70, 280]], dtype=np.int32)
    cv2.fillPoly(img, [pts], (0, 0, 0))

    success, encoded = cv2.imencode(".jpg", img)
    assert success
    img_bytes = encoded.tobytes()

    warped_bytes = warp_perspective_crop(img_bytes)
    assert isinstance(warped_bytes, bytes)
    assert len(warped_bytes) > 0

    # Ensure result is openable as an image
    pil_img = Image.open(io.BytesIO(warped_bytes))
    assert pil_img.size[0] > 0 and pil_img.size[1] > 0
