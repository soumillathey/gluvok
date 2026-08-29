class ANPRServiceError(Exception):
    """Base exception for ANPR domain errors."""

    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class ProviderNotFoundError(ANPRServiceError):
    """Raised when an unknown recognition provider is requested."""

    def __init__(self, provider: str, available_providers: list[str]):
        message = f"Unknown provider '{provider}'. Available providers: {', '.join(available_providers)}"
        super().__init__(message=message, status_code=400)


class InvalidImageError(ANPRServiceError):
    """Raised when an input file is not a valid image."""

    def __init__(self, message: str = "Invalid or unsupported image file"):
        super().__init__(message=message, status_code=400)


class PayloadTooLargeError(ANPRServiceError):
    """Raised when an input exceeds the byte or pixel budget."""

    def __init__(self, message: str = "Image exceeds the maximum permitted size"):
        super().__init__(message=message, status_code=413)
