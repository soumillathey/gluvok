from typing import ClassVar

from app.core.config import settings
from app.core.exceptions import ProviderNotFoundError
from app.core.logging import logger
from app.schemas.plate import ProviderEnum
from app.services.base import BasePlateRecognizer
from app.services.strategies.docling_ocr import DoclingStrategy
from app.services.strategies.nvidia_vision import NvidiaVisionStrategy
from app.services.strategies.plate_recognizer import PlateRecognizerStrategy


class PlateRecognizerFactory:
    """
    Factory Pattern for selecting and instantiating ANPR Model Strategies dynamically.
    """

    _strategies: ClassVar[dict[ProviderEnum, type[BasePlateRecognizer]]] = {
        ProviderEnum.DOCLING: DoclingStrategy,
        ProviderEnum.PLATERECOGNIZER: PlateRecognizerStrategy,
        ProviderEnum.NVIDIA: NvidiaVisionStrategy,
    }

    @classmethod
    def list_providers(cls) -> list[ProviderEnum]:
        return list(cls._strategies.keys())

    @classmethod
    def get_recognizer(cls, provider: ProviderEnum | str | None = None) -> BasePlateRecognizer:
        """
        Factory method to return the appropriate strategy instance.
        If provider is omitted, uses DEFAULT_PROVIDER from settings.
        """
        target_provider = provider or settings.DEFAULT_PROVIDER

        if isinstance(target_provider, str):
            try:
                target_provider = ProviderEnum(target_provider.lower())
            except ValueError:
                logger.warning(f"Requested invalid provider name: '{target_provider}'")
                raise ProviderNotFoundError(
                    provider=target_provider, available_providers=[p.value for p in cls.list_providers()]
                ) from None

        if target_provider not in cls._strategies:
            logger.warning(f"Unregistered strategy requested: '{target_provider}'")
            raise ProviderNotFoundError(
                provider=target_provider.value, available_providers=[p.value for p in cls.list_providers()]
            )

        strategy_cls = cls._strategies[target_provider]
        logger.debug(f"Resolved strategy class '{strategy_cls.__name__}' for provider '{target_provider.value}'.")
        return strategy_cls()
