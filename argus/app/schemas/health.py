from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class HealthStatusEnum(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ComponentHealth(BaseModel):
    status: HealthStatusEnum = Field(..., description="Component status")
    details: str | None = Field(None, description="Diagnostic details or error message")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional component metadata")


class HealthResponse(BaseModel):
    status: HealthStatusEnum = Field(..., description="Overall system health status")
    project_name: str = Field(..., description="Name of the service")
    version: str = Field(..., description="Current running application version")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp in UTC when health check was performed",
    )
    components: dict[str, ComponentHealth] = Field(
        default_factory=dict,
        description="Health status breakdown per component",
    )
