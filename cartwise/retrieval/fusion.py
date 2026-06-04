"""Stage-seven candidate fusion with source-aware filtering and weighted RRF."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any

from cartwise.retrieval.filters import (
    FilterConstraints,
    derive_category_tags,
    derive_color_tags,
    derive_material_tags,
    normalize_string,
)

DENSE_CHANNEL = "dense"
BM25_CHANNEL = "bm25"
LIGHTGCN_CHANNEL = "lightgcn"
POPULARITY_CHANNEL = "popularity"
SEARCH_CHANNELS = frozenset({DENSE_CHANNEL, BM25_CHANNEL})
SOURCE_ORDER = (DENSE_CHANNEL, BM25_CHANNEL, LIGHTGCN_CHANNEL, POPULARITY_CHANNEL)


@dataclass(frozen=True, slots=True)
class FusionConfig:
    """Configurable recall and fusion sizes for the stage-seven main chain."""

    dense_k: int = 30
    bm25_k: int = 30
    lightgcn_k: int = 30
    popularity_k: int = 30
    final_top_k: int = 10
    rrf_k: int = 60


@dataclass(frozen=True, slots=True)
class FusionOutput:
    """Complete fusion artifacts before and after final truncation."""

    final_results: list[dict[str, Any]]
    ranked_results: list[dict[str, Any]]
    filtered_results: list[dict[str, Any]]


def known_user_weights() -> dict[str, float]:
    return {
        DENSE_CHANNEL: 0.45,
        BM25_CHANNEL: 0.25,
        LIGHTGCN_CHANNEL: 0.25,
        POPULARITY_CHANNEL: 0.05,
    }


def cold_start_weights() -> dict[str, float]:
    return {
        DENSE_CHANNEL: 0.65,
        BM25_CHANNEL: 0.30,
        POPULARITY_CHANNEL: 0.05,
    }


def fuse_candidates(
    candidates_by_channel: Mapping[str, Sequence[Mapping[str, Any]]],
    constraints: FilterConstraints,
    *,
    config: FusionConfig = FusionConfig(),
    known_user: bool,
) -> FusionOutput:
    """Merge source candidates, apply source-aware filters, and rank with weighted RRF."""

    _validate_config(config)
    weights = known_user_weights() if known_user else cold_start_weights()
    merged = _merge_candidates(candidates_by_channel)
    kept: list[dict[str, Any]] = []
    filtered: list[dict[str, Any]] = []
    for candidate in merged.values():
        decision = _filter_candidate(candidate, constraints)
        if decision["kept"]:
            kept.append(candidate)
            continue
        filtered.append(_filtered_record(candidate, decision))

    ranked = [
        _ranked_record(candidate, weights, config.rrf_k)
        for candidate in kept
        if any(source in weights for source in candidate["sources"])
    ]
    ranked.sort(
        key=lambda record: (
            -record["fusion_score"],
            min(record["source_ranks"].values()),
            record["parent_asin"],
        )
    )
    for rank, record in enumerate(ranked, start=1):
        record["rank"] = rank
    return FusionOutput(
        final_results=ranked[: config.final_top_k],
        ranked_results=ranked,
        filtered_results=filtered,
    )


def _validate_config(config: FusionConfig) -> None:
    for name in ("dense_k", "bm25_k", "lightgcn_k", "popularity_k", "final_top_k"):
        if getattr(config, name) <= 0:
            raise ValueError(f"{name} must be greater than zero")
    if config.rrf_k < 0:
        raise ValueError("rrf_k must be non-negative")


def _merge_candidates(
    candidates_by_channel: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for channel in SOURCE_ORDER:
        for fallback_rank, candidate in enumerate(
            candidates_by_channel.get(channel, ()),
            start=1,
        ):
            parent_asin = str(candidate.get("parent_asin") or "").strip()
            if not parent_asin:
                continue
            rank = _read_positive_int(candidate.get("rank"), fallback_rank)
            record = merged.setdefault(
                parent_asin,
                {
                    "parent_asin": parent_asin,
                    "item": _candidate_item(candidate, parent_asin),
                    "sources": [],
                    "source_ranks": {},
                    "source_scores": {},
                    "source_score_types": {},
                    "retrieval_queries": {},
                    "documents": {},
                },
            )
            if channel not in record["sources"]:
                record["sources"].append(channel)
            record["source_ranks"][channel] = rank
            record["source_scores"][channel] = candidate.get("score")
            record["source_score_types"][channel] = candidate.get("score_type")
            if candidate.get("retrieval_query") is not None:
                record["retrieval_queries"][channel] = candidate["retrieval_query"]
            if candidate.get("document") is not None:
                record["documents"][channel] = candidate["document"]
            candidate_item = _candidate_item(candidate, parent_asin)
            if len(candidate_item) > len(record["item"]):
                record["item"] = candidate_item
    return merged


def _read_positive_int(value: Any, fallback: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return fallback
    return value


def _candidate_item(candidate: Mapping[str, Any], parent_asin: str) -> dict[str, Any]:
    item = candidate.get("item")
    if isinstance(item, Mapping):
        return dict(item)
    return {"parent_asin": parent_asin}


def _filter_candidate(
    candidate: Mapping[str, Any],
    constraints: FilterConstraints,
) -> dict[str, Any]:
    sources = set(candidate["sources"])
    if sources.intersection(SEARCH_CHANNELS):
        reason = _filter_reason(candidate["item"], _search_constraints(constraints))
        return {
            "kept": reason is None,
            "filter_policy": "search",
            "filter_reason": reason,
        }
    if not constraints.category_tags:
        return {
            "kept": False,
            "filter_policy": "personalized",
            "filter_reason": "missing_category_constraint",
        }
    reason = _filter_reason(candidate["item"], constraints)
    return {
        "kept": reason is None,
        "filter_policy": "personalized",
        "filter_reason": reason,
    }


def _search_constraints(constraints: FilterConstraints) -> FilterConstraints:
    return FilterConstraints(
        min_price=constraints.min_price,
        max_price=constraints.max_price,
        brands=constraints.brands,
        excluded_brands=constraints.excluded_brands,
        category_tags=constraints.category_tags,
    )


def _filter_reason(
    item: Mapping[str, Any], constraints: FilterConstraints
) -> str | None:
    category_tags = _normalize_strings(constraints.category_tags)
    if category_tags:
        item_categories = derive_category_tags(item)
        if not item_categories:
            return "category_missing"
        if not any(
            category_tag in item_category
            for category_tag in category_tags
            for item_category in item_categories
        ):
            return "category_mismatch"

    has_price_constraint = (
        constraints.min_price is not None or constraints.max_price is not None
    )
    price = _read_price(item)
    if has_price_constraint and price is None:
        return "price_missing"
    if constraints.min_price is not None and price < constraints.min_price:
        return "price_below_min"
    if constraints.max_price is not None and price > constraints.max_price:
        return "price_above_max"

    brand = normalize_string(item.get("brand"))
    brands = _normalize_strings(constraints.brands)
    excluded_brands = _normalize_strings(constraints.excluded_brands)
    if brands and brand not in brands:
        return "brand_mismatch"
    if brand in excluded_brands:
        return "brand_excluded"

    color_tags = _normalize_strings(constraints.color_tags)
    if color_tags and not color_tags.intersection(derive_color_tags(item)):
        return "color_mismatch"
    material_tags = _normalize_strings()
    if material_tags and not material_tags.intersection(derive_material_tags(item)):
        return "material_mismatch"
    return None


def _normalize_strings(values: Sequence[str] | Any) -> frozenset[str]:
    return frozenset(
        normalized
        for value in values
        if (normalized := normalize_string(value)) is not None
    )


def _read_price(item: Mapping[str, Any]) -> float | None:
    value = item.get("price")
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    price = float(value)
    return price if math.isfinite(price) else None


def _filtered_record(
    candidate: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "parent_asin": candidate["parent_asin"],
        "sources": list(candidate["sources"]),
        "source_ranks": dict(candidate["source_ranks"]),
        "source_scores": dict(candidate["source_scores"]),
        "source_score_types": dict(candidate["source_score_types"]),
        "filter_policy": decision["filter_policy"],
        "filter_reason": decision["filter_reason"],
        "item": dict(candidate["item"]),
    }


def _ranked_record(
    candidate: Mapping[str, Any],
    weights: Mapping[str, float],
    rrf_k: int,
) -> dict[str, Any]:
    contributions = {
        source: weights[source] / (rrf_k + candidate["source_ranks"][source])
        for source in candidate["sources"]
        if source in weights
    }
    fusion_score = sum(contributions.values())
    return {
        "rank": 0,
        "parent_asin": candidate["parent_asin"],
        "score": fusion_score,
        "score_type": "weighted_rrf",
        "fusion_score": fusion_score,
        "sources": list(candidate["sources"]),
        "source_ranks": dict(candidate["source_ranks"]),
        "source_scores": dict(candidate["source_scores"]),
        "source_score_types": dict(candidate["source_score_types"]),
        "source_weights": {
            source: weights[source]
            for source in candidate["sources"]
            if source in weights
        },
        "rrf_contributions": contributions,
        "retrieval_queries": dict(candidate["retrieval_queries"]),
        "documents": dict(candidate["documents"]),
        "item": dict(candidate["item"]),
        "channel": "fusion",
        "retrieval_query": _first_retrieval_query(candidate["retrieval_queries"]),
    }


def _first_retrieval_query(retrieval_queries: Mapping[str, Any]) -> Any:
    for source in SOURCE_ORDER:
        if source in retrieval_queries:
            return retrieval_queries[source]
    return None
