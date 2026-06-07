"""Application-independent recommendation orchestration."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
import logging
from time import perf_counter
from typing import Any

from cartwise.query.types import FilterConstraints
from cartwise.recommendation.types import RecommendationRequest, RecommendationResult
from cartwise.retrieval.fusion import (
    BM25_CHANNEL,
    DENSE_CHANNEL,
    FusionConfig,
    LIGHTGCN_CHANNEL,
    POPULARITY_CHANNEL,
    fuse_candidates,
)
from cartwise.retrieval.filters import resolve_filter_constraints


FilterResolver = Callable[..., FilterConstraints]
FusionFunction = Callable[..., Any]

logger = logging.getLogger(__name__)


class RecommendationService:
    """Run the current stage-seven recommendation chain with injected resources."""

    def __init__(
        self,
        *,
        intent_parser: Any,
        dense_retriever: Any,
        bm25_retriever: Any,
        popularity_recommender: Any,
        lightgcn_recommender: Any,
        items_by_parent_asin: Mapping[str, Mapping[str, Any]],
        filter_resolver: FilterResolver = resolve_filter_constraints,
        fusion_function: FusionFunction = fuse_candidates,
        fusion_config: FusionConfig = FusionConfig(),
    ) -> None:
        self.intent_parser = intent_parser
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.popularity_recommender = popularity_recommender
        self.lightgcn_recommender = lightgcn_recommender
        self.items_by_parent_asin = items_by_parent_asin
        self.filter_resolver = filter_resolver
        self.fusion_function = fusion_function
        self.fusion_config = fusion_config

    def recommend(self, request: RecommendationRequest) -> RecommendationResult:
        total_started = perf_counter()
        config = (
            replace(self.fusion_config, final_top_k=request.top_k)
            if request.top_k is not None
            else self.fusion_config
        )
        intent_started = perf_counter()
        intent = self.intent_parser.parse(request.query)
        intent_ms = _elapsed_ms(intent_started)
        constraints = self.filter_resolver(
            product_terms=intent.product_terms,
            brands=intent.filters.brands,
            excluded_brands=intent.filters.excluded_brands,
            min_price=intent.filters.min_price,
            max_price=intent.filters.max_price,
            color_tags=intent.filters.color_tags,
            material_tags=intent.filters.material_tags,
        )
        normalized_user_id = request.user_id.strip() if request.user_id else ""
        known_user = bool(
            normalized_user_id
            and normalized_user_id in self.lightgcn_recommender.user_to_index
        )
        personalization_user_id = normalized_user_id or "__cold_start__"
        retrieval_started = perf_counter()
        candidates_by_channel = {
            DENSE_CHANNEL: self._dense_candidates(intent.search_query, config.dense_k),
            BM25_CHANNEL: self._bm25_candidates(intent.search_query, config.bm25_k),
            LIGHTGCN_CHANNEL: (
                self._lightgcn_candidates(normalized_user_id, config.lightgcn_k)
                if known_user
                else []
            ),
            POPULARITY_CHANNEL: self._popularity_candidates(
                personalization_user_id,
                config.popularity_k,
            ),
        }
        retrieval_ms = _elapsed_ms(retrieval_started)
        fusion_started = perf_counter()
        output = self.fusion_function(
            candidates_by_channel,
            constraints,
            config=config,
            known_user=known_user,
        )
        fusion_ms = _elapsed_ms(fusion_started)
        result = RecommendationResult(
            query=request.query,
            search_query=intent.search_query,
            known_user=known_user,
            intent={
                "search_query": intent.search_query,
                "product_terms": list(intent.product_terms),
                "raw_filters": _filter_constraints_payload(intent.filters),
            },
            filter_constraints=constraints,
            filter_constraints_payload=_filter_constraints_payload(constraints),
            candidates_by_channel=candidates_by_channel,
            fusion_output=output,
            final_candidates=output.final_results,
        )
        logger.info(
            "cartwise_timing recommendation_service total_ms=%s intent_parsing_ms=%s "
            "retrieval_ms=%s fusion_ms=%s retrieval_fusion_ms=%s "
            "dense_candidates=%s bm25_candidates=%s lightgcn_candidates=%s "
            "popularity_candidates=%s final_candidates=%s known_user=%s",
            _elapsed_ms(total_started),
            intent_ms,
            retrieval_ms,
            fusion_ms,
            retrieval_ms + fusion_ms,
            len(candidates_by_channel[DENSE_CHANNEL]),
            len(candidates_by_channel[BM25_CHANNEL]),
            len(candidates_by_channel[LIGHTGCN_CHANNEL]),
            len(candidates_by_channel[POPULARITY_CHANNEL]),
            len(result.final_candidates),
            known_user,
        )
        return result

    def _dense_candidates(self, query: str, k: int) -> list[dict[str, Any]]:
        return [
            {
                "channel": DENSE_CHANNEL,
                "rank": rank,
                "parent_asin": result["parent_asin"],
                "score": result["dense_score"],
                "score_type": "dense_score",
                "item": _item(self.items_by_parent_asin, result["parent_asin"]),
                "retrieval_query": result["retrieval_query"],
                "document": result.get("document"),
            }
            for rank, result in enumerate(self.dense_retriever.search(query, k=k), start=1)
        ]

    def _bm25_candidates(self, query: str, k: int) -> list[dict[str, Any]]:
        return [
            {
                "channel": BM25_CHANNEL,
                "rank": rank,
                "parent_asin": result["parent_asin"],
                "score": result["bm25_score"],
                "score_type": "bm25_score",
                "item": _item(self.items_by_parent_asin, result["parent_asin"]),
                "retrieval_query": result["retrieval_query"],
                "document": result["document"],
            }
            for rank, result in enumerate(self.bm25_retriever.search(query, k=k), start=1)
        ]

    def _lightgcn_candidates(self, user_id: str, k: int) -> list[dict[str, Any]]:
        return [
            {
                "channel": LIGHTGCN_CHANNEL,
                "rank": rank,
                "parent_asin": parent_asin,
                "score": None,
                "score_type": None,
                "item": _item(self.items_by_parent_asin, parent_asin),
                "retrieval_query": None,
            }
            for rank, parent_asin in enumerate(
                self.lightgcn_recommender.recommend(user_id, k=k),
                start=1,
            )
        ]

    def _popularity_candidates(self, user_id: str, k: int) -> list[dict[str, Any]]:
        return [
            {
                "channel": POPULARITY_CHANNEL,
                "rank": rank,
                "parent_asin": parent_asin,
                "score": self.popularity_recommender.item_counts[parent_asin],
                "score_type": "interaction_count",
                "item": _item(self.items_by_parent_asin, parent_asin),
                "retrieval_query": None,
            }
            for rank, parent_asin in enumerate(
                self.popularity_recommender.recommend(user_id, k=k),
                start=1,
            )
        ]


def _item(
    items_by_parent_asin: Mapping[str, Mapping[str, Any]],
    parent_asin: str,
) -> dict[str, Any]:
    return dict(items_by_parent_asin.get(parent_asin, {"parent_asin": parent_asin}))


def _filter_constraints_payload(constraints: FilterConstraints) -> dict[str, Any]:
    return {
        "category_tags": list(constraints.category_tags),
        "min_price": constraints.min_price,
        "max_price": constraints.max_price,
        "brands": list(constraints.brands),
        "excluded_brands": list(constraints.excluded_brands),
        "color_tags": list(constraints.color_tags),
        "material_tags": list(constraints.material_tags),
    }


def _elapsed_ms(started: float) -> int:
    return max(0, round((perf_counter() - started) * 1000))
