from __future__ import annotations

from cartwise.application.service import RecommendationApplicationService
from cartwise.application.types import ApplicationRecommendationRequest
from cartwise.evidence.service import EvidenceService
from cartwise.recommendation.service import RecommendationService
from cartwise.retrieval.fusion import FusionConfig
from tests.regression.legacy_harness import (
    FakeBM25Retriever,
    FakeDenseRetriever,
    FakeEvidenceRetriever,
    FakeIntentParser,
    FakeLightGCNRecommender,
    FakePopularityRecommender,
    ITEMS_BY_PARENT_ASIN,
    _patched_resolver,
)


def build_application(*, no_evidence: bool = False) -> RecommendationApplicationService:
    recommendation_service = RecommendationService(
        dense_retriever=FakeDenseRetriever(),
        bm25_retriever=FakeBM25Retriever(),
        lightgcn_recommender=FakeLightGCNRecommender(),
        popularity_recommender=FakePopularityRecommender(),
        items_by_parent_asin=ITEMS_BY_PARENT_ASIN,
        intent_parser=FakeIntentParser(),
        filter_resolver=_patched_resolver,
        fusion_config=FusionConfig(
            dense_k=3,
            bm25_k=3,
            lightgcn_k=3,
            popularity_k=4,
            final_top_k=5,
        ),
    )
    evidence_service = EvidenceService(
        evidence_retriever=FakeEvidenceRetriever(no_evidence=no_evidence),
        generator=None,
    )
    return RecommendationApplicationService(
        recommendation_service=recommendation_service,
        evidence_service=evidence_service,
    )


def test_application_service_combines_recommendations_and_evidence() -> None:
    result = build_application().recommend(
        ApplicationRecommendationRequest(
            query="guitar tuner for beginners",
            top_k=5,
        )
    )

    assert result.query == "guitar tuner for beginners"
    assert result.search_query == "guitar tuner for beginners"
    assert result.applied_constraints["category_tags"] == ["Guitar Tuners"]
    assert [item.parent_asin for item in result.recommendations] == [
        "TUNER_A",
        "TUNER_B",
        "FENDER_A",
        "OVER_BUDGET",
    ]
    assert result.recommendations[0].evidence[0]["review_id"] == "TUNER_A-R1"
    assert result.recommendations[0].fallback is True


def test_application_service_preserves_empty_recommendation_shape() -> None:
    result = build_application().recommend(
        ApplicationRecommendationRequest(query="empty candidate query", top_k=5)
    )

    assert result.recommendations == []
    assert result.recommendation_result.final_candidates == []
    assert result.evidence_result.explanations == []


def test_application_request_does_not_accept_smoke_mode() -> None:
    try:
        ApplicationRecommendationRequest(  # type: ignore[call-arg]
            query="guitar tuner for beginners",
            mode="smoke_search_only",
        )
    except TypeError as error:
        assert "mode" in str(error)
    else:
        raise AssertionError("ApplicationRecommendationRequest accepted smoke-only mode")
