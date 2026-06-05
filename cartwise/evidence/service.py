"""Service wrapper for evidence retrieval and grounded explanations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from cartwise.evidence.rag import EvidenceRagConfig, ProductExplanation, explain_candidates
from cartwise.evidence.types import EvidenceItem, EvidenceRequest, EvidenceResult
from cartwise.recommendation.types import RecommendedCandidate


ExplainFunction = Callable[..., list[ProductExplanation]]


class EvidenceService:
    """Run evidence retrieval and explanation generation for selected candidates."""

    def __init__(
        self,
        *,
        evidence_retriever: Any,
        generator: Any | None = None,
        config: EvidenceRagConfig = EvidenceRagConfig(),
        explain_function: ExplainFunction = explain_candidates,
    ) -> None:
        self.evidence_retriever = evidence_retriever
        self.generator = generator
        self.config = config
        self.explain_function = explain_function

    def explain(self, request: EvidenceRequest) -> EvidenceResult:
        explanations: list[ProductExplanation] = []
        for candidate in request.candidates:
            explanations.extend(
                self.explain_function(
                    english_query=request.english_query,
                    candidates=[_candidate_payload(candidate)],
                    retriever=self.evidence_retriever,
                    generator=self.generator,
                    config=self.config,
                )
            )
        evidence_by_product = {
            explanation.parent_asin: [
                EvidenceItem(
                    parent_asin=evidence.parent_asin,
                    review_id=evidence.review_id,
                    chunk_id=evidence.chunk_id,
                    rating=evidence.rating,
                    text=evidence.text,
                    chunk_text=evidence.chunk_text,
                    score=evidence.score,
                    metadata={
                        "title": evidence.title,
                        "helpful_vote": evidence.helpful_vote,
                        "verified_purchase": evidence.verified_purchase,
                        "timestamp": evidence.timestamp,
                    },
                )
                for evidence in explanation.evidence
            ]
            for explanation in explanations
        }
        return EvidenceResult(
            explanations=explanations,
            evidence_by_product=evidence_by_product,
        )


def _candidate_payload(candidate: RecommendedCandidate) -> Mapping[str, Any]:
    return {
        "parent_asin": candidate.parent_asin,
        "rank": candidate.rank,
        "fusion_score": candidate.fusion_score,
        "sources": list(candidate.sources),
        "source_ranks": dict(candidate.source_ranks),
        "source_scores": dict(candidate.source_scores),
        "item": dict(candidate.item),
        "retrieval_query": candidate.retrieval_query,
        "document": candidate.document,
    }
