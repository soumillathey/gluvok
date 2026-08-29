from fastapi import APIRouter

from app.api.endpoints import health, recognition

api_router = APIRouter()

api_router.include_router(health.router)
api_router.include_router(recognition.router)
