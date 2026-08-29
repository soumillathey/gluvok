from app.services.base import BasePlateRecognizer
from app.services.factory import PlateRecognizerFactory
from app.services.strategies.docling_ocr import DoclingStrategy
from app.services.strategies.nvidia_vision import NvidiaVisionStrategy
from app.services.strategies.plate_recognizer import PlateRecognizerStrategy

__all__ = [
    "BasePlateRecognizer",
    "DoclingStrategy",
    "NvidiaVisionStrategy",
    "PlateRecognizerFactory",
    "PlateRecognizerStrategy",
]
