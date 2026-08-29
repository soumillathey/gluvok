"""
camera_manager.py
Low-resource IP camera frame capture module.
Supports HTTP JPEG Snapshots and on-demand OpenCV RTSP stream frame grabbing.
Uses ThreadPoolExecutor for concurrent multi-camera snapshot capture.
"""

import time
import logging
import requests
from typing import Optional, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..config.camera_config import (
    CAMERA_TIMEOUT,
    MAX_PARALLEL_CAMERA_WORKERS,
    AUXILIARY_CAMERA_URLS,
    ANPR_CAMERA_URL,
)

logger = logging.getLogger(__name__)


def fetch_image_bytes(camera_url: str, timeout: float = CAMERA_TIMEOUT) -> Optional[bytes]:
    """
    Fetch raw JPEG image bytes from a camera URL.
    Prefers HTTP JPEG Snapshots for low CPU/RAM footprint.
    Falls back to on-demand OpenCV RTSP single-frame capture if URL starts with 'rtsp://'.
    """
    if not camera_url:
        return None

    if camera_url.lower().startswith("rtsp://"):
        return _fetch_rtsp_frame(camera_url, timeout)
    else:
        return _fetch_http_snapshot(camera_url, timeout)


def _fetch_http_snapshot(url: str, timeout: float) -> Optional[bytes]:
    """Fetch HTTP JPEG snapshot with low memory footprint."""
    try:
        # verify=False handles cameras with self-signed SSL certificates
        response = requests.get(url, timeout=timeout, verify=False)
        if response.status_code == 200 and response.content:
            return response.content
        else:
            logger.warning(f"[Camera] HTTP fetch failed for {url}: Status {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"[Camera] Exception fetching snapshot from {url}: {e}")
        return None


def _fetch_rtsp_frame(rtsp_url: str, timeout: float) -> Optional[bytes]:
    """On-demand single frame capture from RTSP stream using OpenCV."""
    try:
        import cv2
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            logger.warning(f"[Camera] Unable to open RTSP stream: {rtsp_url}")
            return None

        # Set low buffer size if supported
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ret, frame = cap.read()
        cap.release()

        if ret and frame is not None:
            # Encode frame to JPEG byte buffer
            success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if success:
                return buffer.tobytes()
        logger.warning(f"[Camera] Failed to read frame from RTSP stream: {rtsp_url}")
        return None
    except Exception as e:
        logger.error(f"[Camera] OpenCV RTSP exception for {rtsp_url}: {e}")
        return None


def capture_auxiliary_snapshots(
    camera_urls: Optional[List[str]] = None,
) -> Dict[int, Optional[bytes]]:
    """
    Concurrently captures snapshot images from all auxiliary cameras (Cameras 2 ... N).
    Returns a dict mapping camera_index (2, 3, ...) to raw image bytes or None.
    """
    urls = camera_urls if camera_urls is not None else AUXILIARY_CAMERA_URLS
    results: Dict[int, Optional[bytes]] = {}

    if not urls:
        logger.info("[Camera] No auxiliary cameras configured. Skipping parallel capture.")
        return results

    logger.info(f"[Camera] Triggering parallel snapshot capture for {len(urls)} auxiliary cameras...")
    start_time = time.time()

    workers = min(len(urls), MAX_PARALLEL_CAMERA_WORKERS)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Map future to camera index (starting at index 2)
        future_to_idx = {
            executor.submit(fetch_image_bytes, url): idx + 2
            for idx, url in enumerate(urls)
        }

        for future in as_completed(future_to_idx):
            cam_idx = future_to_idx[future]
            try:
                img_bytes = future.result()
                results[cam_idx] = img_bytes
                status = "SUCCESS" if img_bytes else "FAILED"
                logger.info(f"[Camera] Camera {cam_idx} snapshot capture: {status}")
            except Exception as exc:
                logger.error(f"[Camera] Camera {cam_idx} snapshot generated exception: {exc}")
                results[cam_idx] = None

    elapsed = time.time() - start_time
    logger.info(f"[Camera] Auxiliary parallel capture completed in {elapsed:.2f}s")
    return results
