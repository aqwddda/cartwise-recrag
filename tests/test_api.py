from __future__ import annotations

from dataclasses import dataclass

from fastapi.testclient import TestClient

from cartwise.api.main import create_app
from cartwise.api.schemas import RecommendRequest
from cartwise.application.types import (
    ApplicationRecommendation,
    ApplicationRecommendationRequest,
    ApplicationRecommendationResult,
)
from cartwise.evidence.types import EvidenceResult
from cartwise.query.types import FilterConstraints
from cartwise.recommendation.types import Diagnostic, RecommendationResult
from cartwise.retrieval.fusion import FusionOutput


@dataclass
class FakeApplicationService:
    error: Exception | None = None
    calls: list[ApplicationRecommendationRequest] | None = None

    def recommend(
        self,
        request: ApplicationRecommendationRequest,
    ) -> ApplicationRecommendationResult:
        if self.calls is None:
            self.calls = []
        self.calls.append(request)
        if self.error is not None:
            raise self.error
        return _application_result(request)


def test_health_live_returns_ok() -> None:
    client = TestClient(create_app())

    response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_ready_returns_ready_when_service_is_injected() -> None:
    client = TestClient(create_app(application_service=FakeApplicationService()))

    response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert response.json()["application_service"] == "ready"


def test_health_ready_returns_503_when_service_is_missing() -> None:
    client = TestClient(create_app())

    response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json()["detail"]["status"] == "not_ready"
    assert response.json()["detail"]["application_service"] == "not_initialized"


def test_recommend_returns_structured_result() -> None:
    client = TestClient(create_app(application_service=FakeApplicationService()))

    response = client.post(
        "/api/v1/recommend",
        json={
            "query": "I need a quiet guitar practice solution for an apartment",
            "user_id": "user-1",
            "top_k": 5,
        },
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["query"] == "I need a quiet guitar practice solution for an apartment"
    assert payload["search_query"] == "quiet guitar practice solution"
    assert payload["known_user"] is True
    assert payload["applied_constraints"] == {"category_tags": ["Guitar Amplifiers"]}
    assert payload["results"][0] == {
        "product_id": "ASIN_1",
        "parent_asin": "ASIN_1",
        "title": "Practice Headphone Amp",
        "brand": "CartWise",
        "price": 19.99,
        "rank": 1,
        "fusion_score": 0.123,
        "sources": ["dense", "bm25"],
        "source_ranks": {"dense": 1, "bm25": 2},
        "source_scores": {"dense": 0.88, "bm25": 3.5},
        "reason": "Matches apartment practice needs.",
        "potential_cons": "It is not a full speaker amp.",
        "fallback": False,
        "evidence": [
            {
                "review_id": "R1",
                "chunk_id": "R1-0",
                "rating": 5.0,
                "text": "Works well for quiet practice.",
                "chunk_text": "Works well for quiet practice.",
                "score": 0.88,
                "metadata": {"verified_purchase": True},
            }
        ],
    }
    assert payload["diagnostics"] == [
        {
            "component": "evidence",
            "error_type": "fallback",
            "message": "template fallback used for one candidate",
            "recoverable": True,
        }
    ]
    assert isinstance(payload["latency_ms"], int)


def test_recommend_passes_request_fields_to_application_service() -> None:
    service = FakeApplicationService()
    client = TestClient(create_app(application_service=service))

    response = client.post(
        "/api/v1/recommend",
        json={
            "query": "  portable microphone stand  ",
            "user_id": "  user-2  ",
            "top_k": 7,
        },
    )

    assert response.status_code == 200
    assert service.calls == [
        ApplicationRecommendationRequest(
            query="portable microphone stand",
            user_id="user-2",
            top_k=7,
        )
    ]


def test_recommend_does_not_expose_internal_application_objects() -> None:
    client = TestClient(create_app(application_service=FakeApplicationService()))

    response = client.post("/api/v1/recommend", json={"query": "guitar tuner"})

    payload = response.json()
    assert response.status_code == 200
    assert "recommendation_result" not in payload
    assert "evidence_result" not in payload
    assert "fusion_output" not in payload
    assert "candidates_by_channel" not in payload


def test_blank_query_returns_validation_error() -> None:
    client = TestClient(create_app(application_service=FakeApplicationService()))

    response = client.post("/api/v1/recommend", json={"query": "   "})

    assert response.status_code == 422


def test_invalid_top_k_returns_validation_error() -> None:
    client = TestClient(create_app(application_service=FakeApplicationService()))

    too_small = client.post(
        "/api/v1/recommend",
        json={"query": "guitar tuner", "top_k": 0},
    )
    too_large = client.post(
        "/api/v1/recommend",
        json={"query": "guitar tuner", "top_k": 51},
    )

    assert too_small.status_code == 422
    assert too_large.status_code == 422


def test_recommend_returns_503_when_service_is_missing() -> None:
    client = TestClient(create_app())

    response = client.post("/api/v1/recommend", json={"query": "guitar tuner"})

    assert response.status_code == 503
    assert "not initialized" in response.json()["detail"]


def test_recommend_returns_controlled_error_when_service_fails() -> None:
    service = FakeApplicationService(error=RuntimeError("qdrant unavailable"))
    client = TestClient(create_app(application_service=service))

    response = client.post("/api/v1/recommend", json={"query": "guitar tuner"})

    assert response.status_code == 503
    assert response.json()["detail"] == "recommendation service is unavailable"


def test_api_request_schema_does_not_expose_internal_modes() -> None:
    assert "mode" not in RecommendRequest.model_fields
    assert "smoke_search_only" not in RecommendRequest.model_fields

    client = TestClient(create_app(application_service=FakeApplicationService()))
    response = client.post(
        "/api/v1/recommend",
        json={"query": "guitar tuner", "mode": "smoke_search_only"},
    )

    assert response.status_code == 422


def _application_result(
    request: ApplicationRecommendationRequest,
) -> ApplicationRecommendationResult:
    diagnostic = Diagnostic(
        component="evidence",
        error_type="fallback",
        message="template fallback used for one candidate",
        recoverable=True,
    )
    return ApplicationRecommendationResult(
        query=request.query,
        search_query="quiet guitar practice solution",
        known_user=request.user_id is not None,
        applied_constraints={"category_tags": ["Guitar Amplifiers"]},
        recommendations=(
            ApplicationRecommendation(
                parent_asin="ASIN_1",
                title="Practice Headphone Amp",
                brand="CartWise",
                price=19.99,
                rank=1,
                fusion_score=0.123,
                sources=("dense", "bm25"),
                source_ranks={"dense": 1, "bm25": 2},
                source_scores={"dense": 0.88, "bm25": 3.5},
                reason="Matches apartment practice needs.",
                potential_cons="It is not a full speaker amp.",
                evidence=(
                    {
                        "review_id": "R1",
                        "chunk_id": "R1-0",
                        "rating": 5.0,
                        "text": "Works well for quiet practice.",
                        "chunk_text": "Works well for quiet practice.",
                        "score": 0.88,
                        "metadata": {"verified_purchase": True},
                    },
                ),
                fallback=False,
            ),
        ),
        recommendation_result=RecommendationResult(
            query=request.query,
            search_query="quiet guitar practice solution",
            known_user=request.user_id is not None,
            intent={},
            filter_constraints=FilterConstraints(),
            filter_constraints_payload={"category_tags": ["Guitar Amplifiers"]},
            candidates_by_channel={"dense": ()},
            fusion_output=FusionOutput(
                final_results=[],
                ranked_results=[],
                filtered_results=[],
            ),
            final_candidates=[],
            diagnostics=(diagnostic,),
        ),
        evidence_result=EvidenceResult(
            explanations=(),
            evidence_by_product={},
            diagnostics=(),
        ),
        diagnostics=(diagnostic,),
    )
