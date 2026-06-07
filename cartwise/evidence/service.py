"""Service wrapper for evidence retrieval and grounded explanations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import logging
from time import perf_counter
from typing import Any

from cartwise.evidence.rag import EvidenceRagConfig, ProductExplanation, explain_candidates
from cartwise.evidence.types import EvidenceItem, EvidenceRequest, EvidenceResult
from cartwise.recommendation.types import RecommendedCandidate


ExplainFunction = Callable[..., list[ProductExplanation]]

logger = logging.getLogger(__name__)


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
        started = perf_counter()
        explanations = self.explain_function(
            english_query=request.english_query,
            candidates=[_candidate_payload(candidate) for candidate in request.candidates],
            retriever=self.evidence_retriever,
            generator=self.generator,
            config=self.config,
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
        result = EvidenceResult(
            explanations=explanations,
            evidence_by_product=evidence_by_product,
        )
        logger.info(
            "cartwise_timing evidence_service total_ms=%s candidates=%s "
            "explanations=%s evidence_items=%s",
            _elapsed_ms(started),
            len(request.candidates),
            len(result.explanations),
            sum(len(items) for items in result.evidence_by_product.values()),
        )
        return result


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


def _elapsed_ms(started: float) -> int:
    return max(0, round((perf_counter() - started) * 1000))
