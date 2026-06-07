from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from cartwise.core.evidence_rag import (
    EvidenceRagConfig,
    QdrantReviewEvidenceRetriever,
    build_review_query,
    explain_candidates,
    retrieve_product_evidence,
)


def candidate(parent_asin: str = "P1") -> dict[str, object]:
    return {
        "parent_asin": parent_asin,
        "retrieval_query": "guitar tuner",
        "item": {
            "parent_asin": parent_asin,
            "title": "Clip-On Guitar Tuner",
            "brand": "Acme",
            "categories": ["Instrument Accessories", "Tuners"],
        },
    }


def hit(
    review_id: str,
    *,
    parent_asin: str = "P1",
    chunk_index: int = 0,
    rating: float = 5.0,
) -> dict[str, object]:
    return {
        "parent_asin": parent_asin,
        "review_id": review_id,
        "chunk_id": f"{review_id}#chunk_{chunk_index}",
        "rating": rating,
        "title": f"title {review_id}",
        "text": f"full review {review_id}",
        "chunk_text": f"chunk text {review_id}-{chunk_index}",
        "helpful_vote": 1,
        "verified_purchase": True,
        "timestamp": 10,
        "score": 0.9,
    }


class FakeRetriever:
    def __init__(self, batches: list[list[dict[str, object]]]) -> None:
        self.batches = batches
        self.calls: list[dict[str, object]] = []

    def search(self, query, *, parent_asin, k, rating_lte=None):
        self.calls.append(
            {
                "query": query,
                "parent_asin": parent_asin,
                "k": k,
                "rating_lte": rating_lte,
            }
        )
        return self.batches.pop(0)


class FakeGenerator:
    def __init__(self, content: str) -> None:
        self.content = content
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.content


@dataclass
class FakeQdrantPoint:
    payload: dict[str, object]
    score: float


@dataclass
class FakeQdrantResponse:
    points: list[FakeQdrantPoint]


class FakeQdrantClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def query_points(self, **kwargs):
        self.calls.append(kwargs)
        parent_asin = kwargs["query_filter"]["must"][0]["match"]["value"]
        return FakeQdrantResponse(
            points=[
                FakeQdrantPoint(
                    payload=hit("r1", parent_asin=parent_asin),
                    score=0.8,
                )
            ]
        )


class CountingEncoder:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def encode_query(self, query: str) -> np.ndarray:
        self.queries.append(query)
        return np.array([float(len(self.queries)), 0.0], dtype=np.float32)


def test_review_query_uses_english_query_title_and_categories() -> None:
    query = build_review_query("beginner tuner", candidate())

    assert "beginner tuner" in query
    assert "Clip-On Guitar Tuner" in query
    assert "Instrument Accessories | Tuners" in query


def test_retrieve_evidence_expands_chunks_and_keeps_duplicate_review_chunks() -> None:
    retriever = FakeRetriever(
        [
            [hit("r1", chunk_index=0), hit("r1", chunk_index=1)],
            [
                hit("r1", chunk_index=0),
                hit("r1", chunk_index=1),
                hit("r2", rating=2),
            ],
        ]
    )

    evidence = retrieve_product_evidence(
        english_query="beginner tuner",
        candidate=candidate(),
        retriever=retriever,
        config=EvidenceRagConfig(initial_chunk_k=2, final_review_k=2, max_candidate_chunk_k=3),
    )

    assert [entry.review_id for entry in evidence] == ["r1", "r1", "r2"]
    assert [call["k"] for call in retriever.calls] == [2, 3]


def test_retrieve_evidence_supplements_low_rating_review() -> None:
    retriever = FakeRetriever(
        [
            [hit("r1", rating=5), hit("r2", rating=4)],
            [hit("r1", rating=5), hit("r2", rating=4), hit("r3", rating=5)],
            [hit("r_bad", rating=2)],
        ]
    )

    evidence = retrieve_product_evidence(
        english_query="beginner tuner",
        candidate=candidate(),
        retriever=retriever,
        config=EvidenceRagConfig(initial_chunk_k=2, final_review_k=3, max_candidate_chunk_k=3),
    )

    assert [entry.review_id for entry in evidence] == ["r1", "r2", "r_bad"]
    assert retriever.calls[-1]["rating_lte"] == 3


def test_valid_llm_explanation_must_cite_retrieved_review_ids() -> None:
    generator = FakeGenerator(
        json.dumps(
            {
                "items": [
                    {
                        "parent_asin": "P1",
                        "reason": "适合初学者使用。",
                        "potential_cons": "可能需要关注安装体验。",
                        "cited_review_ids": ["r1"],
                    }
                ]
            }
        )
    )
    explanations = explain_candidates(
        english_query="beginner tuner",
        candidates=[candidate()],
        retriever=FakeRetriever([[hit("r1", rating=5), hit("r2", rating=2)]]),
        generator=generator,
        config=EvidenceRagConfig(initial_chunk_k=2, final_review_k=2, max_candidate_chunk_k=2),
    )

    assert explanations[0].fallback is False
    assert explanations[0].cited_review_ids == ("r1",)
    assert "chunk text r1-0" in generator.prompts[0]


def test_invalid_llm_citation_falls_back_to_template() -> None:
    generator = FakeGenerator(
        json.dumps(
            {
                "items": [
                    {
                        "parent_asin": "P1",
                        "reason": "bad",
                        "potential_cons": "bad",
                        "cited_review_ids": ["missing"],
                    }
                ]
            }
        )
    )
    explanations = explain_candidates(
        english_query="beginner tuner",
        candidates=[candidate()],
        retriever=FakeRetriever([[hit("r1", rating=2)]]),
        generator=generator,
        config=EvidenceRagConfig(initial_chunk_k=1, final_review_k=1, max_candidate_chunk_k=1),
    )

    assert explanations[0].fallback is True
    assert explanations[0].cited_review_ids == ("r1",)
    assert "review_id: r1" in explanations[0].potential_cons


def test_retrieval_outside_candidate_scope_falls_back_without_evidence() -> None:
    explanations = explain_candidates(
        english_query="beginner tuner",
        candidates=[candidate("P1")],
        retriever=FakeRetriever([[hit("r1", parent_asin="P2")]]),
    )

    assert explanations[0].fallback is True
    assert explanations[0].parent_asin == "P1"
    assert explanations[0].evidence == ()


def test_qdrant_evidence_retriever_reuses_query_embedding() -> None:
    client = FakeQdrantClient()
    encoder = CountingEncoder()
    retriever = QdrantReviewEvidenceRetriever(
        client,
        collection="reviews",
        encoder=encoder,
    )

    retriever.search("same review query", parent_asin="P1", k=2)
    retriever.search("same review query", parent_asin="P1", k=3, rating_lte=3)
    retriever.search("different review query", parent_asin="P1", k=2)

    assert encoder.queries == ["same review query", "different review query"]
    assert client.calls[0]["query"] == [1.0, 0.0]
    assert client.calls[1]["query"] == [1.0, 0.0]
    assert client.calls[2]["query"] == [2.0, 0.0]
