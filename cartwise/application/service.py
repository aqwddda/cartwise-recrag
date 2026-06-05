"""Thin application service combining recommendation and evidence services."""

from __future__ import annotations

from typing import Any

from cartwise.application.types import (
    ApplicationRecommendation,
    ApplicationRecommendationRequest,
    ApplicationRecommendationResult,
)
from cartwise.evidence.types import EvidenceRequest
from cartwise.recommendation.types import (
    RecommendationRequest,
    recommended_candidate_from_mapping,
)


class RecommendationApplicationService:
    """Call recommendation first, then evidence, and shape final results."""

    def __init__(self, *, recommendation_service: Any, evidence_service: Any) -> None:
        self.recommendation_service = recommendation_service
        self.evidence_service = evidence_service

    def recommend(
        self,
        request: ApplicationRecommendationRequest,
    ) -> ApplicationRecommendationResult:
        recommendation_result = self.recommendation_service.recommend(
            RecommendationRequest(
                query=request.query,
                user_id=request.user_id,
                top_k=request.top_k,
                mode=request.mode,
            )
        )
        candidates = tuple(
            recommended_candidate_from_mapping(candidate)
            for candidate in recommendation_result.final_candidates
        )
        evidence_result = self.evidence_service.explain(
            EvidenceRequest(
                query=request.query,
                english_query=recommendation_result.search_query,
                candidates=candidates,
            )
        )
        explanations_by_parent_asin = {
            explanation.parent_asin: explanation
            for explanation in evidence_result.explanations
        }
        recommendations = []
        for candidate in recommendation_result.final_candidates:
            item = candidate.get("item", {})
            explanation = explanations_by_parent_asin.get(candidate["parent_asin"])
            evidence = [] if explanation is None else [
                {
                    "review_id": entry.review_id,
                    "chunk_id": entry.chunk_id,
                    "rating": entry.rating,
                    "text": entry.text,
                    "chunk_text": entry.chunk_text,
                    "score": entry.score,
                }
                for entry in explanation.evidence
            ]
            recommendations.append(
                ApplicationRecommendation(
                    parent_asin=candidate["parent_asin"],
                    title=item.get("title"),
                    brand=item.get("brand"),
                    price=item.get("price"),
                    rank=candidate["rank"],
                    fusion_score=candidate["fusion_score"],
                    sources=list(candidate.get("sources", ())),
                    source_ranks=dict(candidate.get("source_ranks", {})),
                    source_scores=dict(candidate.get("source_scores", {})),
                    reason="" if explanation is None else explanation.reason,
                    potential_cons=(
                        "" if explanation is None else explanation.potential_cons
                    ),
                    evidence=evidence,
                    fallback=True if explanation is None else explanation.fallback,
                )
            )
        return ApplicationRecommendationResult(
            query=request.query,
            search_query=recommendation_result.search_query,
            known_user=recommendation_result.known_user,
            applied_constraints=recommendation_result.filter_constraints_payload,
            recommendations=recommendations,
            recommendation_result=recommendation_result,
            evidence_result=evidence_result,
            diagnostics=(
                *recommendation_result.diagnostics,
                *evidence_result.diagnostics,
            ),
        )
