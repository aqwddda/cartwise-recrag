from __future__ import annotations

import pytest

from cartwise.retrieval.filters import FilterConstraints
from cartwise.retrieval.fusion import FusionConfig, fuse_candidates


def candidate(
    parent_asin: str,
    *,
    rank: int,
    score: float | None = None,
    title: str | None = None,
    price: float = 20.0,
    brand: str = "Acme",
    categories: list[str] | None = None,
) -> dict[str, object]:
    return {
        "rank": rank,
        "parent_asin": parent_asin,
        "score": score,
        "score_type": "test_score",
        "retrieval_query": "guitar tuner",
        "item": {
            "parent_asin": parent_asin,
            "title": title or parent_asin,
            "price": price,
            "brand": brand,
            "categories": categories or [],
        },
    }


def test_weighted_rrf_merges_sources_and_prefers_multi_source_items() -> None:
    output = fuse_candidates(
        {
            "dense": [
                candidate("A", rank=1, score=0.9),
                candidate("B", rank=2, score=0.8),
            ],
            "bm25": [
                candidate("B", rank=1, score=12.0),
                candidate("C", rank=2, score=8.0),
            ],
        },
        FilterConstraints(),
        config=FusionConfig(final_top_k=10),
        known_user=False,
    )

    assert [record["parent_asin"] for record in output.final_results] == ["B", "A", "C"]
    top = output.final_results[0]
    assert top["sources"] == ["dense", "bm25"]
    assert top["source_ranks"] == {"dense": 2, "bm25": 1}
    assert top["source_scores"] == {"dense": 0.8, "bm25": 12.0}
    assert top["fusion_score"] == pytest.approx(0.65 / 62 + 0.30 / 61)


def test_search_source_uses_conservative_filter_when_item_also_personalized() -> None:
    output = fuse_candidates(
        {
            "dense": [candidate("A", rank=1, categories=[])],
            "lightgcn": [candidate("A", rank=1, categories=[])],
        },
        FilterConstraints(category_tags=("Accessories",)),
        config=FusionConfig(final_top_k=10),
        known_user=True,
    )

    assert [record["parent_asin"] for record in output.final_results] == ["A"]
    assert output.filtered_results == []


def test_personalized_only_candidates_need_mapped_category_constraint() -> None:
    output = fuse_candidates(
        {
            "lightgcn": [candidate("A", rank=1, categories=["General Accessories"])],
            "popularity": [candidate("B", rank=1, categories=["General Accessories"])],
        },
        FilterConstraints(),
        config=FusionConfig(final_top_k=10),
        known_user=True,
    )

    assert output.final_results == []
    assert [record["parent_asin"] for record in output.filtered_results] == ["A", "B"]
    assert {record["filter_reason"] for record in output.filtered_results} == {
        "missing_category_constraint"
    }


def test_personalized_only_candidates_use_full_filters_and_report_reason() -> None:
    output = fuse_candidates(
        {
            "lightgcn": [
                candidate("KEEP", rank=1, categories=["General Accessories"]),
                candidate("DROP", rank=2, categories=["Guitars"]),
            ],
        },
        FilterConstraints(category_tags=("Accessories",), max_price=30.0),
        config=FusionConfig(final_top_k=10),
        known_user=True,
    )

    assert [record["parent_asin"] for record in output.final_results] == ["KEEP"]
    assert output.filtered_results[0]["parent_asin"] == "DROP"
    assert output.filtered_results[0]["filter_policy"] == "personalized"
    assert output.filtered_results[0]["filter_reason"] == "category_mismatch"


def test_final_top_k_truncates_ranked_results_without_dropping_full_sequence() -> None:
    output = fuse_candidates(
        {
            "dense": [
                candidate("A", rank=1),
                candidate("B", rank=2),
                candidate("C", rank=3),
            ],
        },
        FilterConstraints(),
        config=FusionConfig(final_top_k=2),
        known_user=False,
    )

    assert [record["parent_asin"] for record in output.final_results] == ["A", "B"]
    assert [record["parent_asin"] for record in output.ranked_results] == ["A", "B", "C"]


def test_invalid_fusion_config_is_rejected() -> None:
    with pytest.raises(ValueError, match="final_top_k"):
        fuse_candidates(
            {"dense": [candidate("A", rank=1)]},
            FilterConstraints(),
            config=FusionConfig(final_top_k=0),
            known_user=False,
        )
