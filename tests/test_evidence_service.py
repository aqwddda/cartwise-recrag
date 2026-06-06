from __future__ import annotations

from typing import Any

from cartwise.evidence.rag import EvidenceRagConfig, ProductExplanation
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


def candidate_for(parent_asin: str, rank: int) -> RecommendedCandidate:
    item = ITEMS_BY_PARENT_ASIN[parent_asin]
    return RecommendedCandidate(
        parent_asin=parent_asin,
        rank=rank,
        fusion_score=0.1 / rank,
        sources=("dense",),
        source_ranks={"dense": rank},
        source_scores={"dense": 1.0 / rank},
        item=item,
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


def test_evidence_service_batches_multiple_candidates_once() -> None:
    calls: list[list[str]] = []

    def fake_explain_function(**kwargs: Any) -> list[ProductExplanation]:
        payloads = kwargs["candidates"]
        calls.append([payload["parent_asin"] for payload in payloads])
        return [
            ProductExplanation(
                parent_asin=payload["parent_asin"],
                reason=f"reason {payload['parent_asin']}",
                potential_cons=f"cons {payload['parent_asin']}",
                cited_review_ids=(),
                evidence=(),
                fallback=True,
            )
            for payload in payloads
        ]

    service = EvidenceService(
        evidence_retriever=FakeEvidenceRetriever(),
        generator=None,
        explain_function=fake_explain_function,
    )
    candidates = (
        candidate_for("TUNER_A", 1),
        candidate_for("TUNER_B", 2),
        candidate_for("FENDER_A", 3),
    )

    result = service.explain(
        EvidenceRequest(
            query="guitar tuner for beginners",
            english_query="guitar tuner for beginners",
            candidates=candidates,
        )
    )

    assert calls == [["TUNER_A", "TUNER_B", "FENDER_A"]]
    assert [item.parent_asin for item in result.explanations] == [
        "TUNER_A",
        "TUNER_B",
        "FENDER_A",
    ]
    assert result.explanations[1].reason == "reason TUNER_B"
