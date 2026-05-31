from __future__ import annotations

import math

import pytest

from cartwise.retrieval.popularity import (
    PopularityRecommender,
    evaluate_recommender,
)


def test_recommend_returns_popular_unseen_items_with_stable_tie_breaking() -> None:
    recommender = PopularityRecommender(
        [
            {"user_id": "U1", "parent_asin": "P1"},
            {"user_id": "U2", "parent_asin": "P1"},
            {"user_id": "U2", "parent_asin": "P2"},
            {"user_id": "U3", "parent_asin": "P2"},
            {"user_id": "U3", "parent_asin": "P3"},
            {"user_id": "U4", "parent_asin": "P4"},
        ]
    )

    assert recommender.recommend("U1", k=3) == ["P2", "P3", "P4"]
    assert recommender.recommend("new-user", k=3) == ["P1", "P2", "P3"]
    assert recommender.recommend("new-user", k=3, excluded_items={"P1"}) == [
        "P2",
        "P3",
        "P4",
    ]


def test_recommend_rejects_negative_k() -> None:
    recommender = PopularityRecommender([])

    with pytest.raises(ValueError, match="non-negative"):
        recommender.recommend("U1", k=-1)


def test_recommend_returns_empty_list_for_zero_k() -> None:
    recommender = PopularityRecommender(
        [{"user_id": "U1", "parent_asin": "P1"}]
    )

    assert recommender.recommend("new-user", k=0) == []


def test_evaluate_recommender_calculates_ranking_metrics() -> None:
    recommender = PopularityRecommender(
        [
            {"user_id": "U1", "parent_asin": "P1"},
            {"user_id": "U3", "parent_asin": "P1"},
            {"user_id": "U2", "parent_asin": "P2"},
            {"user_id": "U3", "parent_asin": "P3"},
        ]
    )

    metrics = evaluate_recommender(
        recommender,
        [
            {"user_id": "U1", "parent_asin": "P2"},
            {"user_id": "U2", "parent_asin": "P3"},
        ],
        k=10,
    )

    assert metrics.users == 2
    assert metrics.recall == 1.0
    assert metrics.hit_rate == 1.0
    assert metrics.ndcg == pytest.approx((1.0 + 1.0 / math.log2(3)) / 2)


def test_evaluate_recommender_excludes_additional_history() -> None:
    recommender = PopularityRecommender(
        [
            {"user_id": "U1", "parent_asin": "P1"},
            {"user_id": "U2", "parent_asin": "P1"},
            {"user_id": "U2", "parent_asin": "P2"},
            {"user_id": "U3", "parent_asin": "P3"},
        ]
    )

    metrics = evaluate_recommender(
        recommender,
        [{"user_id": "U1", "parent_asin": "P3"}],
        k=1,
        additional_history=[{"user_id": "U1", "parent_asin": "P2"}],
    )

    assert metrics.recall == 1.0
    assert metrics.ndcg == 1.0
    assert metrics.hit_rate == 1.0
