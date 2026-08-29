from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class APIErrorResponse(BaseModel):
    success: bool = Field(False, description="Always False for error responses")
    status_code: int = Field(..., description="HTTP status code")
    message: str = Field(..., description="Human-readable error description")
    error_type: str = Field(..., description="Exception class or category")
    details: Any = Field(None, description="Detailed validation or contextual error info")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp of the error",
    )
