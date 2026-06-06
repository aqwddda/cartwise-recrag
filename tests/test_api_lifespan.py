from __future__ import annotations

from dataclasses import dataclass

from fastapi.testclient import TestClient

from cartwise.api.main import create_app
from cartwise.application.types import ApplicationRecommendationRequest
from tests.test_api import FakeApplicationService


@dataclass
class CountingBuilder:
    service: FakeApplicationService | None = None
    error: Exception | None = None
    calls: int = 0

    def __call__(self):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.service or FakeApplicationService()


def test_create_app_fake_service_still_bypasses_startup_builder() -> None:
    builder = CountingBuilder(error=RuntimeError("builder should not run"))
    service = FakeApplicationService()
    app = create_app(
        application_service=service,
        application_service_builder=builder,
    )

    with TestClient(app) as client:
        response = client.get("/health/ready")

    assert response.status_code == 200
    assert builder.calls == 0


def test_default_startup_builder_success_makes_app_ready() -> None:
    builder = CountingBuilder()
    app = create_app(application_service_builder=builder)

    with TestClient(app) as client:
        ready = client.get("/health/ready")

    assert builder.calls == 1
    assert ready.status_code == 200
    assert ready.json()["application_service"] == "ready"
    assert ready.json()["resources"]["qdrant"] == "ready"


def test_builder_failure_returns_not_ready_and_does_not_retry_per_request() -> None:
    builder = CountingBuilder(error=RuntimeError("missing qdrant collection"))
    app = create_app(application_service_builder=builder)

    with TestClient(app) as client:
        first_ready = client.get("/health/ready")
        second_ready = client.get("/health/ready")
        recommend = client.post("/api/v1/recommend", json={"query": "guitar tuner"})

    assert builder.calls == 1
    assert first_ready.status_code == 503
    assert second_ready.status_code == 503
    assert first_ready.json()["detail"]["application_service"] == "initialization_failed"
    assert "missing qdrant collection" in first_ready.json()["detail"][
        "initialization_error"
    ]
    assert recommend.status_code == 503
    assert "initialization failed" in recommend.json()["detail"]


def test_recommend_uses_startup_constructed_application_service() -> None:
    service = FakeApplicationService()
    app = create_app(application_service_builder=CountingBuilder(service=service))

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/recommend",
            json={"query": "guitar tuner", "user_id": "user-1", "top_k": 3},
        )

    assert response.status_code == 200
    assert service.calls == [
        ApplicationRecommendationRequest(
            query="guitar tuner",
            user_id="user-1",
            top_k=3,
        )
    ]
