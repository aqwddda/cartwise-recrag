from __future__ import annotations

from cartwise.evidence.service import EvidenceService
from cartwise.retrieval.fusion import FusionConfig
from scripts.tools.stage8_smoke_adapter import Stage8SmokeAdapter
from tests.regression.legacy_harness import (
    FakeBM25Retriever,
    FakeDenseRetriever,
    FakeEvidenceRetriever,
    ITEMS_BY_PARENT_ASIN,
)


def test_stage8_smoke_adapter_preserves_search_only_flow() -> None:
    evidence_service = EvidenceService(
        evidence_retriever=FakeEvidenceRetriever(no_evidence=True),
        generator=None,
    )
    adapter = Stage8SmokeAdapter(
        dense_retriever=FakeDenseRetriever(),
        bm25_retriever=FakeBM25Retriever(),
        evidence_service=evidence_service,
        items_by_parent_asin=ITEMS_BY_PARENT_ASIN,
        fusion_config=FusionConfig(dense_k=3, bm25_k=3, final_top_k=2),
    )

    result = adapter.run(query="guitar tuner for beginners", top_k=2)

    assert [item["parent_asin"] for item in result.candidates_by_channel["dense"]] == [
        "TUNER_A",
        "TUNER_B",
        "FENDER_A",
    ]
    assert [item["parent_asin"] for item in result.candidates_by_channel["bm25"]] == [
        "TUNER_B",
        "TUNER_A",
        "OVER_BUDGET",
    ]
    assert [item["parent_asin"] for item in result.fusion_output.final_results] == [
        "TUNER_A",
        "TUNER_B",
    ]
    assert [item.parent_asin for item in result.final_candidates] == [
        "TUNER_A",
        "TUNER_B",
    ]
    assert [item.parent_asin for item in result.evidence_result.explanations] == [
        "TUNER_A",
        "TUNER_B",
    ]
