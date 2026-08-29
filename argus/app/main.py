import argparse
import json
import sys

from app.core.config import settings
from app.core.contracts import ContractViolation
from app.core.exceptions import ANPRServiceError
from app.core.logging import logger
from app.services.pipeline import recognize_plate_image
from app.services.strategies.docling_ocr import check_docling_engine
from app.services.yolo_filter import get_yolo_model


def main():
    parser = argparse.ArgumentParser(description=f"{settings.PROJECT_NAME} Runner")
    parser.add_argument(
        "image",
        nargs="?",
        default=None,
        help="Optional path to image file for single-image CLI plate recognition",
    )
    parser.add_argument(
        "--server",
        action="store_true",
        help="Start the FastAPI REST API server (default when no image argument is provided)",
    )
    parser.add_argument(
        "--host",
        default=settings.SERVER_HOST,
        help=f"Host to bind the server to (default: {settings.SERVER_HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=settings.SERVER_PORT,
        help=f"Port to bind the server to (default: {settings.SERVER_PORT})",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="Recognition provider override (docling, nvidia, platerecognizer)",
    )

    args = parser.parse_args()

    # If no image is provided, or if --server is explicitly specified, run the FastAPI server
    if args.server or args.image is None:
        import uvicorn

        logger.info(f"Starting {settings.PROJECT_NAME} FastAPI server on {args.host}:{args.port}...")
        uvicorn.run("app.server:app", host=args.host, port=args.port, reload=False)
        return

    # Warm YOLO model & check OCR engines
    try:
        get_yolo_model()
        logger.info("YOLO v11 model loaded successfully.")
    except (RuntimeError, ValueError, OSError, AttributeError) as e:
        logger.warning(f"Warning loading YOLO model: {e}")

    check_docling_engine()

    try:
        response = recognize_plate_image(args.image, provider=args.provider)
        print(json.dumps(response.model_dump(), indent=2))
    except (ANPRServiceError, ContractViolation, ValueError, OSError, RuntimeError) as e:
        logger.error(f"Error processing image '{args.image}': {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
