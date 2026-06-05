from __future__ import annotations

import math

from cartwise.recommendation.service import RecommendationService
from cartwise.recommendation.types import RecommendationRequest
from cartwise.retrieval.fusion import FusionConfig
from tests.regression.legacy_harness import (
    FakeBM25Retriever,
    FakeDenseRetriever,
    FakeIntentParser,
    FakeLightGCNRecommender,
    FakePopularityRecommender,
    ITEMS_BY_PARENT_ASIN,
    _patched_resolver,
)


def build_service() -> RecommendationService:
    return RecommendationService(
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


def test_recommendation_service_preserves_cold_start_fusion_order() -> None:
    result = build_service().recommend(
        RecommendationRequest(query="guitar tuner for beginners", top_k=5)
    )

    assert result.search_query == "guitar tuner for beginners"
    assert result.known_user is False
    assert result.intent["product_terms"] == ["guitar tuner"]
    assert result.filter_constraints_payload["category_tags"] == ["Guitar Tuners"]
    assert [item["parent_asin"] for item in result.final_candidates] == [
        "TUNER_A",
        "TUNER_B",
        "FENDER_A",
        "OVER_BUDGET",
    ]
    assert math.isclose(
        result.final_candidates[0]["fusion_score"],
        0.01631411951348493,
        rel_tol=0.0,
        abs_tol=1e-6,
    )


def test_recommendation_service_preserves_known_user_lightgcn_branch() -> None:
    result = build_service().recommend(
        RecommendationRequest(
            query="guitar tuner for beginners",
            user_id="known-user",
            top_k=5,
        )
    )

    assert result.known_user is True
    assert result.candidates_by_channel["lightgcn"]
    assert result.final_candidates[0]["sources"] == [
        "dense",
        "bm25",
        "lightgcn",
        "popularity",
    ]


def test_recommendation_service_preserves_empty_candidate_shape() -> None:
    result = build_service().recommend(
        RecommendationRequest(query="empty candidate query", top_k=5)
    )

    assert result.final_candidates == []
    assert result.fusion_output.ranked_results == []
    assert [item["parent_asin"] for item in result.fusion_output.filtered_results] == [
        "TUNER_A",
        "STAND_A",
        "MIC_A",
        "OVER_BUDGET",
    ]


def test_recommendation_request_does_not_accept_smoke_mode() -> None:
    try:
        RecommendationRequest(  # type: ignore[call-arg]
            query="guitar tuner for beginners",
            mode="smoke_search_only",
        )
    except TypeError as error:
        assert "mode" in str(error)
    else:
        raise AssertionError("RecommendationRequest accepted smoke-only mode")
