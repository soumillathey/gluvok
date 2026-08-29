from app.core.config import settings
from app.schemas.health import HealthResponse


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == settings.PROJECT_NAME
    assert data["version"] == settings.VERSION
    assert data["status"] == "running"
    assert data["docs"] == "/docs"
    assert data["health"] == "/health"


def test_health_endpoint_success(client):
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    validated = HealthResponse.model_validate(data)

    assert validated.status in ["healthy", "degraded"]
    assert validated.project_name == settings.PROJECT_NAME
    assert validated.version == settings.VERSION
    assert "yolo" in validated.components
    assert "docling" in validated.components
    assert "providers" in validated.components
    assert validated.components["yolo"].status in ["healthy", "degraded"]
    assert validated.components["docling"].status == "healthy"


def test_process_time_header(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert "x-process-time-ms" in response.headers
    process_time = float(response.headers["x-process-time-ms"])
    assert process_time >= 0.0
