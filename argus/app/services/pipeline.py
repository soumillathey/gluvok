import time
from typing import Any

from pydantic import ValidationError
from requests.exceptions import RequestException

from app.core.config import settings
from app.core.contracts import ContractViolation
from app.core.exceptions import (
    ANPRServiceError,
    InvalidImageError,
    PayloadTooLargeError,
)
from app.core.logging import logger
from app.schemas.plate import (
    BoundingBox,
    ImageInput,
    PlateResult,
    ProviderEnum,
    RecognitionResponse,
    RecognitionStatusEnum,
)
from app.services.factory import PlateRecognizerFactory
from app.services.image_processing import decode_and_downscale
from app.services.yolo_filter import filter_vehicle_and_occupancy

_FIXED_FALLBACK_ORDER = [
    ProviderEnum.DOCLING,
    ProviderEnum.NVIDIA,
    ProviderEnum.PLATERECOGNIZER,
]


def get_fallback_providers() -> list[ProviderEnum]:
    return [
        settings.DEFAULT_PROVIDER,
        *[p for p in _FIXED_FALLBACK_ORDER if p != settings.DEFAULT_PROVIDER],
    ]


def validate_plate_results(
    raw_results: Any,
    provider: ProviderEnum,
) -> list[PlateResult]:
    if not isinstance(raw_results, list):
        logger.error(f"Provider '{provider.value}' returned {type(raw_results).__name__}, expected list.")
        return []

    validated: list[PlateResult] = []
    for item in raw_results:
        if not isinstance(item, dict):
            logger.warning(f"Provider '{provider.value}' returned non-dict result ({type(item).__name__}); discarding.")
            continue
        try:
            validated.append(PlateResult.model_validate(item))
        except ValidationError as exc:
            logger.warning(
                f"Provider '{provider.value}' returned unusable result "
                f"{ {k: item.get(k) for k in list(item)[:4]} }: {exc.error_count()} error(s); discarding."
            )
    return validated


def run_waterfall(
    image_bytes: bytes,
    filename: str,
    vehicle_box: BoundingBox | None = None,
    vehicle_boxes: list[BoundingBox] | None = None,
    override_provider: ProviderEnum | None = None,
) -> tuple[list[PlateResult], ProviderEnum]:
    providers = [override_provider] if override_provider else get_fallback_providers()

    for provider in providers:
        logger.info(f"Attempting recognition on '{filename}' using provider: '{provider.value}'")
        try:
            recognizer = PlateRecognizerFactory.get_recognizer(provider)
            raw_results = recognizer.recognize(
                image_bytes,
                filename=filename,
                vehicle_box=vehicle_box,
                vehicle_boxes=vehicle_boxes,
            )
        except ContractViolation:
            raise
        except (ANPRServiceError, RequestException, ValueError, RuntimeError, OSError, KeyError, AttributeError) as exc:
            logger.error(f"Provider '{provider.value}' encountered an error: {exc}. Falling back to next provider...")
            continue

        plate_results = validate_plate_results(raw_results, provider)
        if plate_results:
            logger.info(
                f"Successfully recognized {len(plate_results)} plate(s) in '{filename}' via '{provider.value}'."
            )
            return plate_results, provider

        logger.warning(
            f"Provider '{provider.value}' returned no usable license plate results. Falling back to next provider..."
        )

    return [], providers[0]


def recognize_plate_image(
    image_input: ImageInput,
    filename: str = "image.jpg",
    provider: ProviderEnum | str | None = None,
) -> RecognitionResponse:
    """
    Main ANPR entry point for processing an image input (file path or bytes).
    Performs YOLO v11 pre-screening and recognition waterfall.
    """
    start_time = time.time()

    if isinstance(image_input, str):
        filename = filename or image_input
        try:
            with open(image_input, "rb") as f:
                raw_bytes = f.read()
        except Exception as e:
            raise InvalidImageError(f"Failed to read image file '{image_input}': {e}") from e
    elif isinstance(image_input, bytes):
        raw_bytes = image_input
    else:
        raise InvalidImageError(f"Unsupported image input type: {type(image_input).__name__}")

    if len(raw_bytes) > settings.MAX_UPLOAD_BYTES:
        raise PayloadTooLargeError(
            f"Image exceeds maximum permitted size of {settings.MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
        )
    if not raw_bytes:
        raise InvalidImageError(f"Image '{filename}' is empty.")

    image_bytes = decode_and_downscale(raw_bytes)

    # Step 1: YOLO Pre-screening
    yolo_result = filter_vehicle_and_occupancy(image_bytes)

    target_provider: ProviderEnum | None = None
    if provider:
        # ProviderEnum subclasses str, so this also covers an already-ProviderEnum input.
        target_provider = ProviderEnum(provider.lower())

    fallback_providers = get_fallback_providers()
    default_prov = target_provider or fallback_providers[0]

    if not yolo_result["is_eligible"]:
        execution_time_ms = round((time.time() - start_time) * 1000, 2)
        logger.info(f"Image '{filename}' ineligible: {yolo_result['status_message']} ({execution_time_ms} ms)")
        return RecognitionResponse(
            success=False,
            rejected=True,
            status=yolo_result["status"],
            status_message=yolo_result["status_message"],
            vehicle_detected=yolo_result["vehicle_detected"],
            vehicle_type=yolo_result["vehicle_type"],
            human_detected=yolo_result["human_detected"],
            filename=filename,
            provider=default_prov,
            results=[],
            execution_time_ms=execution_time_ms,
        )

    # Step 2: Recognition Waterfall across all detected vehicle boxes
    plate_results, active_provider = run_waterfall(
        image_bytes,
        filename=filename,
        vehicle_box=yolo_result.get("vehicle_box"),
        vehicle_boxes=yolo_result.get("vehicle_boxes"),
        override_provider=target_provider,
    )

    execution_time_ms = round((time.time() - start_time) * 1000, 2)
    has_valid_plate = any(r.plate != "N/A" for r in plate_results)
    final_status = RecognitionStatusEnum.SUCCESS if has_valid_plate else RecognitionStatusEnum.NO_PLATE_DETECTED

    if has_valid_plate:
        target_name = yolo_result["vehicle_type"] or ("vehicle" if yolo_result["vehicle_detected"] else None)
        if target_name:
            status_msg = (
                f"License plate successfully detected and recognized on "
                f"{target_name} via {active_provider.value}."
            )
        else:
            status_msg = f"License plate successfully detected and recognized via {active_provider.value}."
    else:
        if yolo_result["vehicle_detected"]:
            status_msg = (
                f"4-wheeler ({yolo_result['vehicle_type'] or 'vehicle'}) detected, "
                f"but no readable license plate characters could be recognized."
            )
        else:
            status_msg = "No vehicle detected and no readable license plate characters could be recognized."

    return RecognitionResponse(
        success=has_valid_plate,
        rejected=False,
        status=final_status,
        status_message=status_msg,
        vehicle_detected=yolo_result["vehicle_detected"],
        vehicle_type=yolo_result["vehicle_type"],
        human_detected=yolo_result["human_detected"],
        filename=filename,
        provider=active_provider,
        results=plate_results,
        execution_time_ms=execution_time_ms,
    )
