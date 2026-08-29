import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.router import api_router
from app.core.config import settings
from app.core.contracts import ContractViolation
from app.core.exceptions import ANPRServiceError
from app.core.logging import logger
from app.schemas.error import APIErrorResponse
from app.services.factory import PlateRecognizerFactory
from app.services.strategies.docling_ocr import check_docling_engine
from app.services.yolo_filter import get_yolo_model


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager:
    Pre-warms YOLO model and verifies OCR engines during service startup.
    """
    logger.info(f"Starting {settings.PROJECT_NAME} v{settings.VERSION}...")

    try:
        get_yolo_model()
        logger.info("YOLO v11 model loaded and warmed successfully.")
    except (RuntimeError, ValueError, OSError, AttributeError) as exc:
        logger.warning(f"Non-fatal warning warming YOLO model during startup: {exc}")

    try:
        check_docling_engine()
        logger.info("Docling OCR engine verified successfully.")
    except (RuntimeError, ValueError, OSError, AttributeError, ImportError) as exc:
        logger.warning(f"Non-fatal warning checking Docling engine during startup: {exc}")

    registered = [p.value for p in PlateRecognizerFactory.list_providers()]
    logger.info(f"Active recognition providers: {registered} (Default: '{settings.DEFAULT_PROVIDER.value}')")

    yield

    logger.info(f"Shutting down {settings.PROJECT_NAME}...")


def create_app() -> FastAPI:
    """FastAPI application factory."""
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description=(
            "Enterprise Automatic Number Plate Recognition (ANPR) Microservice powered by YOLO v11 and RapidOCR."
        ),
        docs_url=settings.DOCS_URL,
        redoc_url=settings.REDOC_URL,
        lifespan=lifespan,
    )

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
        allow_methods=settings.CORS_ALLOW_METHODS,
        allow_headers=settings.CORS_ALLOW_HEADERS,
    )

    # Process time header middleware
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start_time = time.perf_counter()
        response = await call_next(request)
        process_time_ms = (time.perf_counter() - start_time) * 1000
        response.headers["X-Process-Time-Ms"] = f"{process_time_ms:.2f}"
        return response

    # Exception Handlers
    @app.exception_handler(ANPRServiceError)
    async def handle_anpr_service_error(request: Request, exc: ANPRServiceError) -> JSONResponse:
        logger.warning(f"Domain exception on {request.url.path}: {exc.message} ({exc.status_code})")
        payload = APIErrorResponse(
            success=False,
            status_code=exc.status_code,
            message=exc.message,
            error_type=exc.__class__.__name__,
            timestamp=datetime.now(UTC),
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=payload.model_dump(mode="json"),
        )

    @app.exception_handler(ContractViolation)
    async def handle_contract_violation(request: Request, exc: ContractViolation) -> JSONResponse:
        logger.error(f"Internal contract violation on {request.url.path}: {exc}")
        payload = APIErrorResponse(
            success=False,
            status_code=500,
            message="Internal system assertion contract failed.",
            error_type="ContractViolation",
            details=str(exc),
            timestamp=datetime.now(UTC),
        )
        return JSONResponse(
            status_code=500,
            content=payload.model_dump(mode="json"),
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        logger.warning(f"Request validation error on {request.url.path}: {exc.errors()}")
        payload = APIErrorResponse(
            success=False,
            status_code=422,
            message="Request validation error.",
            error_type="RequestValidationError",
            details=exc.errors(),
            timestamp=datetime.now(UTC),
        )
        return JSONResponse(
            status_code=422,
            content=payload.model_dump(mode="json"),
        )

    @app.exception_handler(Exception)
    async def handle_unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(f"Unhandled server error on {request.url.path}: {exc}")
        payload = APIErrorResponse(
            success=False,
            status_code=500,
            message="An internal server error occurred.",
            error_type="InternalServerError",
            timestamp=datetime.now(UTC),
        )
        return JSONResponse(
            status_code=500,
            content=payload.model_dump(mode="json"),
        )

    # Root endpoint for service information
    @app.get(
        "/",
        summary="Service Information",
        description="Root metadata information for the Argus ANPR API.",
        tags=["Info"],
    )
    async def root():
        return {
            "name": settings.PROJECT_NAME,
            "version": settings.VERSION,
            "status": "running",
            "docs": "/docs",
            "health": "/health",
        }

    # Mount API routers
    app.include_router(api_router)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.server:app",
        host=settings.SERVER_HOST,
        port=settings.SERVER_PORT,
        reload=True,
    )
