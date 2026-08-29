from fastapi import APIRouter, File, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.core.config import settings
from app.core.exceptions import InvalidImageError, PayloadTooLargeError
from app.schemas.plate import RecognitionResponse
from app.services.pipeline import recognize_plate_image

router = APIRouter()


@router.post(
    "/recognize",
    response_model=RecognitionResponse,
    summary="Recognize Vehicle License Plate",
    description=(
        "Upload a vehicle image (JPEG, PNG, WebP) to execute YOLO v11 pre-screening "
        "and automated license plate recognition waterfall."
    ),
    tags=["Recognition"],
)
async def recognize_plate(
    file: UploadFile | None = File(None, description="Image file (JPEG, PNG, WebP)"),  # noqa: B008
    image: UploadFile | None = File(None, description="Image file alias (JPEG, PNG, WebP)"),  # noqa: B008
) -> RecognitionResponse:
    uploaded = file or image
    if uploaded is None:
        raise InvalidImageError("Uploaded image file is missing (expected 'file' or 'image' field).")

    if uploaded.size is not None and uploaded.size > settings.MAX_UPLOAD_BYTES:
        raise PayloadTooLargeError(
            f"Image exceeds maximum permitted size of {settings.MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
        )

    image_bytes = await uploaded.read()

    if not image_bytes:
        raise InvalidImageError("Uploaded image file is empty.")

    if len(image_bytes) > settings.MAX_UPLOAD_BYTES:
        raise PayloadTooLargeError(
            f"Image exceeds maximum permitted size of {settings.MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
        )

    filename = uploaded.filename or "uploaded_image.jpg"

    # Run heavy CV / OCR inference in worker threadpool to prevent blocking asyncio loop
    response: RecognitionResponse = await run_in_threadpool(
        recognize_plate_image,
        image_input=image_bytes,
        filename=filename,
    )

    return response
