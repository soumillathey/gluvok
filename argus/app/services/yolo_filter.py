import threading
from typing import TypedDict

import numpy as np
from PIL import Image
from ultralytics import YOLO

from app.core.config import settings
from app.core.contracts import bounded, ensure, require
from app.core.logging import logger
from app.schemas.plate import BoundingBox, ImageInput, RecognitionStatusEnum
from app.services.image_processing import clamp_box, load_rgb


class YoloResult(TypedDict, total=True):
    """Typed return value of filter_vehicle_and_occupancy."""

    is_eligible: bool
    status: RecognitionStatusEnum | None
    status_message: str
    vehicle_detected: bool
    vehicle_type: str | None
    human_detected: bool
    vehicle_box: BoundingBox | None
    vehicle_boxes: list[BoundingBox]
    vehicle_count: int


# Global lazy-loaded YOLO model instance, guarded by a thread lock.
_YOLO_MODEL: YOLO | None = None
_YOLO_LOCK = threading.Lock()


def get_yolo_model() -> YOLO:
    global _YOLO_MODEL
    if _YOLO_MODEL is None:
        with _YOLO_LOCK:
            if _YOLO_MODEL is None:
                target_model = (
                    settings.YOLO_MODEL_NAME
                    if settings.YOLO_MODEL_NAME and "11" in settings.YOLO_MODEL_NAME
                    else "yolo11n.pt"
                )
                logger.debug(f"Loading YOLO model weights: {target_model}")
                _YOLO_MODEL = YOLO(target_model)
    ensure(_YOLO_MODEL is not None, "YOLO model failed to initialise")
    return _YOLO_MODEL


# COCO Class Names for 4-wheelers
PERSON_CLASS_ID = 0
FOUR_WHEELER_CLASS_NAMES = {2: "car", 5: "bus", 7: "truck"}

# Upper bound on detections examined per frame.
MAX_DETECTIONS = 100


def _run_detection(
    pil_img: Image.Image,
    human_conf_thresh: float,
    vehicle_conf_thresh: float,
) -> tuple[bool, list[tuple[str, BoundingBox]]]:
    """
    Run YOLO and extract (human_present, area-sorted vehicles).
    Every coordinate is clamped to the real frame before leaving this function.
    """
    require(pil_img is not None, "_run_detection called with no image")

    width, height = pil_img.size
    results = next(iter(get_yolo_model()(pil_img, verbose=False)))

    human_detected = False
    vehicles: list[tuple[int, str, BoundingBox]] = []

    boxes = getattr(results, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return False, []

    if hasattr(boxes, "cls") and hasattr(boxes.cls, "cpu"):
        cls_ids = boxes.cls.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy() if getattr(boxes, "xyxy", None) is not None else None
    elif hasattr(boxes, "cls"):
        cls_ids = np.array(boxes.cls)
        confs = np.array(boxes.conf)
        xyxy = np.array(boxes.xyxy) if getattr(boxes, "xyxy", None) is not None else None
    else:
        return False, []

    # Bounded: a frame with hundreds of detections is a frame we decline to fully process
    detections = bounded(list(zip(cls_ids, confs, strict=False)), MAX_DETECTIONS, "YOLO detections")

    for idx, (raw_cls, conf) in enumerate(detections):
        cls_id = int(raw_cls)
        if cls_id == PERSON_CLASS_ID and conf >= human_conf_thresh:
            human_detected = True
            continue

        if cls_id not in FOUR_WHEELER_CLASS_NAMES or conf < vehicle_conf_thresh:
            continue

        raw_box = None
        if xyxy is not None and idx < len(xyxy) and len(xyxy[idx]) >= 4:
            xb = xyxy[idx]
            raw_box = (int(xb[0]), int(xb[1]), int(xb[2]), int(xb[3]))
        box = clamp_box(raw_box, width, height)
        if box is None:
            logger.warning(
                f"[yolo] discarding unusable {FOUR_WHEELER_CLASS_NAMES[cls_id]} box {raw_box} "
                f"for a {width}x{height} frame."
            )
            continue

        area = (box[2] - box[0]) * (box[3] - box[1])
        vehicles.append((area, FOUR_WHEELER_CLASS_NAMES[cls_id], box))

    # Largest first: at a weighbridge the vehicle on the platform dominates the frame
    vehicles.sort(key=lambda item: item[0], reverse=True)
    return human_detected, [(v_type, box) for _, v_type, box in vehicles]


def filter_vehicle_and_occupancy(
    image_input: ImageInput,
    human_conf_thresh: float | None = None,
    vehicle_conf_thresh: float | None = None,
    reject_on_human: bool | None = None,
    reject_on_multiple_vehicles: bool | None = None,
    reject_on_no_vehicle: bool | None = None,
) -> YoloResult:
    """
    Pre-screening gate combining vehicle detection and occupancy filtering.
    """
    human_conf_thresh = human_conf_thresh or settings.HUMAN_CONF_THRESH
    vehicle_conf_thresh = vehicle_conf_thresh or settings.VEHICLE_CONF_THRESH
    reject_on_human = reject_on_human if reject_on_human is not None else settings.REJECT_ON_HUMAN_DETECTED
    reject_on_multiple_vehicles = (
        reject_on_multiple_vehicles if reject_on_multiple_vehicles is not None else settings.REJECT_ON_MULTIPLE_VEHICLES
    )
    reject_on_no_vehicle = (
        reject_on_no_vehicle if reject_on_no_vehicle is not None else settings.REJECT_ON_NO_VEHICLE
    )

    require(
        0.0 <= human_conf_thresh <= 1.0,
        f"human_conf_thresh must be in [0, 1], got {human_conf_thresh}",
    )
    require(
        0.0 <= vehicle_conf_thresh <= 1.0,
        f"vehicle_conf_thresh must be in [0, 1], got {vehicle_conf_thresh}",
    )

    pil_img = load_rgb(image_input)
    human_detected, vehicles = _run_detection(pil_img, human_conf_thresh, vehicle_conf_thresh)

    detected_vehicle_types = [v_type for v_type, _ in vehicles]
    vehicle_boxes = [box for _, box in vehicles]
    vehicle_count = len(vehicles)

    primary_vehicle_type = detected_vehicle_types[0] if detected_vehicle_types else None
    primary_vehicle_box = vehicle_boxes[0] if vehicle_boxes else None

    if human_detected and reject_on_human:
        logger.warning("Rejected frame: Human presence detected.")
        return {
            "is_eligible": False,
            "status": RecognitionStatusEnum.REJECTED_HUMAN_DETECTED,
            "status_message": "Image rejected: Human presence detected.",
            "vehicle_detected": vehicle_count > 0,
            "vehicle_type": primary_vehicle_type,
            "human_detected": human_detected,
            "vehicle_box": primary_vehicle_box,
            "vehicle_boxes": vehicle_boxes,
            "vehicle_count": vehicle_count,
        }

    if vehicle_count > 1 and reject_on_multiple_vehicles:
        types_str = ", ".join(detected_vehicle_types)
        logger.warning(f"Rejected frame: {vehicle_count} vehicles detected ({types_str}).")
        return {
            "is_eligible": False,
            "status": RecognitionStatusEnum.REJECTED_MULTIPLE_VEHICLES,
            "status_message": f"Image rejected: Multiple 4-wheeler vehicles detected ({vehicle_count} vehicles: {types_str}). Weighbridge allows only 1 vehicle.",
            "vehicle_detected": True,
            "vehicle_type": primary_vehicle_type,
            "human_detected": human_detected,
            "vehicle_box": primary_vehicle_box,
            "vehicle_boxes": vehicle_boxes,
            "vehicle_count": vehicle_count,
        }

    if vehicle_count == 0:
        if reject_on_no_vehicle:
            return {
                "is_eligible": False,
                "status": RecognitionStatusEnum.REJECTED_NO_FOUR_WHEELER,
                "status_message": "Image rejected: No 4-wheeler vehicle (car, bus, truck) detected.",
                "vehicle_detected": False,
                "vehicle_type": None,
                "human_detected": human_detected,
                "vehicle_box": None,
                "vehicle_boxes": [],
                "vehicle_count": 0,
            }
        occupancy_note = "with human presence" if human_detected else "with no human occupancy"
        return {
            "is_eligible": True,
            "status": None,
            "status_message": f"No vehicle detected ({occupancy_note}). Eligible for direct plate recognition.",
            "vehicle_detected": False,
            "vehicle_type": None,
            "human_detected": human_detected,
            "vehicle_box": None,
            "vehicle_boxes": [],
            "vehicle_count": 0,
        }

    occupancy_note = "with human presence" if human_detected else "with no human occupancy"
    multi_note = f" ({vehicle_count} vehicles detected)" if vehicle_count > 1 else ""
    return {
        "is_eligible": True,
        "status": None,
        "status_message": f"4-wheeler ({primary_vehicle_type}){multi_note} detected {occupancy_note}. Eligible for plate recognition.",
        "vehicle_detected": True,
        "vehicle_type": primary_vehicle_type,
        "human_detected": human_detected,
        "vehicle_box": primary_vehicle_box,
        "vehicle_boxes": vehicle_boxes,
        "vehicle_count": vehicle_count,
    }
