from __future__ import annotations

from cartwise.evidence.rag import EvidenceRagConfig
from cartwise.evidence.service import EvidenceService
from cartwise.evidence.types import EvidenceRequest
from cartwise.recommendation.types import RecommendedCandidate
from tests.regression.legacy_harness import FakeEvidenceRetriever, ITEMS_BY_PARENT_ASIN


def candidate() -> RecommendedCandidate:
    return RecommendedCandidate(
        parent_asin="TUNER_A",
        rank=1,
        fusion_score=0.1,
        sources=("dense",),
        source_ranks={"dense": 1},
        source_scores={"dense": 0.9},
        item=ITEMS_BY_PARENT_ASIN["TUNER_A"],
    )


def test_evidence_service_returns_template_explanation_with_evidence() -> None:
    service = EvidenceService(
        evidence_retriever=FakeEvidenceRetriever(),
        generator=None,
        config=EvidenceRagConfig(initial_chunk_k=2, final_review_k=3, max_candidate_chunk_k=3),
    )

    result = service.explain(
        EvidenceRequest(
            query="guitar tuner for beginners",
            english_query="guitar tuner for beginners",
            candidates=(candidate(),),
        )
    )

    explanation = result.explanations[0]
    assert explanation.parent_asin == "TUNER_A"
    assert explanation.fallback is True
    assert list(explanation.cited_review_ids) == [
        "TUNER_A-R1",
        "TUNER_A-R2",
        "TUNER_A-LOW",
    ]
    assert [item.review_id for item in result.evidence_by_product["TUNER_A"]] == [
        "TUNER_A-R1",
        "TUNER_A-R2",
        "TUNER_A-LOW",
    ]


def test_evidence_service_preserves_no_evidence_fallback() -> None:
    service = EvidenceService(
        evidence_retriever=FakeEvidenceRetriever(no_evidence=True),
        generator=None,
    )

    result = service.explain(
        EvidenceRequest(
            query="guitar tuner for beginners",
            english_query="guitar tuner for beginners",
            candidates=(candidate(),),
        )
    )

    explanation = result.explanations[0]
    assert explanation.parent_asin == "TUNER_A"
    assert explanation.fallback is True
    assert explanation.cited_review_ids == ()
    assert result.evidence_by_product["TUNER_A"] == []
