"""Compatibility adapter for the historical stage-eight smoke flow.

This module is intentionally script-scoped. The production CartWise services must
not import it; it preserves the old search-only smoke behavior while formal
application services always run the full recommendation chain.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from cartwise.evidence.types import EvidenceRequest, EvidenceResult
from cartwise.query.types import FilterConstraints
from cartwise.recommendation.types import (
    RecommendedCandidate,
    recommended_candidate_from_mapping,
)
from cartwise.retrieval.fusion import (
    BM25_CHANNEL,
    DENSE_CHANNEL,
    FusionConfig,
    FusionOutput,
    fuse_candidates,
)


FusionFunction = Callable[..., FusionOutput]


@dataclass(frozen=True, slots=True)
class Stage8SmokeResult:
    query: str
    candidates_by_channel: Mapping[str, list[dict[str, Any]]]
    fusion_output: FusionOutput
    evidence_result: EvidenceResult
    final_candidates: tuple[RecommendedCandidate, ...] = field(default_factory=tuple)


class Stage8SmokeAdapter:
    """Run the old Dense/BM25-only stage-eight smoke chain."""

    def __init__(
        self,
        *,
        dense_retriever: Any,
        bm25_retriever: Any,
        evidence_service: Any,
        items_by_parent_asin: Mapping[str, Mapping[str, Any]],
        fusion_config: FusionConfig = FusionConfig(),
        fusion_function: FusionFunction = fuse_candidates,
    ) -> None:
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.evidence_service = evidence_service
        self.items_by_parent_asin = items_by_parent_asin
        self.fusion_config = fusion_config
        self.fusion_function = fusion_function

    def run(self, *, query: str, top_k: int | None = None) -> Stage8SmokeResult:
        config = (
            self._replace_final_top_k(top_k)
            if top_k is not None
            else self.fusion_config
        )
        constraints = FilterConstraints()
        candidates_by_channel = {
            DENSE_CHANNEL: [
                candidate
                for candidate in self._dense_candidates(query, config.dense_k)
                if candidate["parent_asin"] in self.items_by_parent_asin
            ],
            BM25_CHANNEL: [
                candidate
                for candidate in self._bm25_candidates(query, config.bm25_k)
                if candidate["parent_asin"] in self.items_by_parent_asin
            ],
        }
        fusion_output = self.fusion_function(
            candidates_by_channel,
            constraints,
            config=config,
            known_user=False,
        )
        final_candidates = tuple(
            recommended_candidate_from_mapping(candidate)
            for candidate in fusion_output.final_results
        )
        evidence_result = self.evidence_service.explain(
            EvidenceRequest(
                query=query,
                english_query=query,
                candidates=final_candidates,
            )
        )
        return Stage8SmokeResult(
            query=query,
            candidates_by_channel=candidates_by_channel,
            fusion_output=fusion_output,
            evidence_result=evidence_result,
            final_candidates=final_candidates,
        )

    def _replace_final_top_k(self, top_k: int) -> FusionConfig:
        return FusionConfig(
            dense_k=self.fusion_config.dense_k,
            bm25_k=self.fusion_config.bm25_k,
            lightgcn_k=self.fusion_config.lightgcn_k,
            popularity_k=self.fusion_config.popularity_k,
            final_top_k=top_k,
            rrf_k=self.fusion_config.rrf_k,
        )

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


def _item(
    items_by_parent_asin: Mapping[str, Mapping[str, Any]],
    parent_asin: str,
) -> dict[str, Any]:
    return dict(items_by_parent_asin.get(parent_asin, {"parent_asin": parent_asin}))
