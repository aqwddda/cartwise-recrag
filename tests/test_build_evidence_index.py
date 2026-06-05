from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.pipeline.build_evidence_index import (
    add_documents_to_vector_store,
    build_review_documents,
    build_review_text,
    collection_name,
    qdrant_payload,
    review_point_id,
    summarize_token_lengths,
    upsert_documents_to_qdrant,
)


REVIEWS = [
    {
        "review_id": "rvw_a",
        "parent_asin": "P1",
        "rating": 5.0,
        "title": "Great",
        "text": "Works well for beginners.",
        "helpful_vote": 3,
        "verified_purchase": True,
        "timestamp": 10,
    },
    {
        "review_id": "rvw_b",
        "parent_asin": "P2",
        "rating": 2.0,
        "title": "Hard",
        "text": "The setup was difficult. The manual was unclear.",
        "helpful_vote": 1,
        "verified_purchase": False,
        "timestamp": 20,
    },
]


class FakeVectorStore:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def add_documents(self, documents, *, ids):
        self.calls.append({"documents": documents, "ids": ids})


class FakeEmbeddings:
    def encode(
        self,
        texts,
        *,
        batch_size,
        normalize_embeddings,
        convert_to_numpy,
        show_progress_bar,
    ):
        return [
            SimpleNamespace(tolist=lambda index=index: [float(index), 1.0])
            for index, _ in enumerate(texts)
        ]


class FakeQdrantClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def upsert(self, *, collection_name, points, wait):
        self.calls.append(
            {
                "collection_name": collection_name,
                "points": points,
                "wait": wait,
            }
        )


def test_collection_and_point_ids_are_stable() -> None:
    assert collection_name("dev", "intfloat/e5-small-v2") == (
        "cartwise_review_evidence_dev_intfloat_e5_small_v2"
    )
    assert review_point_id("rvw_a#chunk_0") == review_point_id("rvw_a#chunk_0")
    assert review_point_id("rvw_a#chunk_0") != review_point_id("rvw_a#chunk_1")


def test_review_text_includes_title_rating_and_body() -> None:
    text = build_review_text(REVIEWS[0])

    assert "Review title: Great" in text
    assert "Rating: 5.0" in text
    assert "Review text: Works well" in text


def test_build_review_documents_keeps_multiple_chunks_for_one_review() -> None:
    documents, report = build_review_documents(
        REVIEWS,
        split_text=lambda text: [text[:12], text[12:]] if "difficult" in text else [text],
    )

    assert report == {"split_reviews": 1, "skipped_reviews": 0}
    assert [document.metadata["chunk_id"] for document in documents] == [
        "rvw_a#chunk_0",
        "rvw_b#chunk_0",
        "rvw_b#chunk_1",
    ]
    assert documents[1].metadata["review_id"] == documents[2].metadata["review_id"]
    assert documents[1].metadata["parent_asin"] == "P2"
    assert documents[1].page_content.startswith("passage: ")


def test_token_length_summary_records_percentiles() -> None:
    stats = summarize_token_lengths([1, 3, 6, 10])

    assert stats.p50 == 3
    assert stats.p90 == 10
    assert stats.p99 == 10


def test_add_documents_to_vector_store_batches_with_stable_ids() -> None:
    documents, _ = build_review_documents(REVIEWS, split_text=lambda text: [text])
    vector_store = FakeVectorStore()

    added = add_documents_to_vector_store(vector_store, documents, batch_size=1)

    assert added == 2
    assert len(vector_store.calls) == 2
    assert vector_store.calls[0]["ids"] == [documents[0].id]
    assert vector_store.calls[1]["ids"] == [documents[1].id]


def test_add_documents_to_vector_store_validates_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        add_documents_to_vector_store(SimpleNamespace(), [], batch_size=0)


def test_qdrant_payload_keeps_minimum_retrieval_metadata() -> None:
    documents, _ = build_review_documents(REVIEWS[:1], split_text=lambda text: [text])
    payload = qdrant_payload(documents[0])

    assert payload == {
        "parent_asin": "P1",
        "review_id": "rvw_a",
        "chunk_id": "rvw_a#chunk_0",
        "rating": 5.0,
        "title": "Great",
        "text": "Works well for beginners.",
        "chunk_text": documents[0].metadata["chunk_text"],
        "helpful_vote": 3,
        "verified_purchase": True,
        "timestamp": 10,
    }


def test_upsert_documents_to_qdrant_batches_points() -> None:
    documents, _ = build_review_documents(REVIEWS, split_text=lambda text: [text])
    client = FakeQdrantClient()

    added = upsert_documents_to_qdrant(
        client=client,
        collection="reviews",
        embeddings=FakeEmbeddings(),
        documents=documents,
        batch_size=1,
    )

    assert added == 2
    assert len(client.calls) == 1
    assert client.calls[0]["collection_name"] == "reviews"
    assert client.calls[0]["wait"] is False
    assert client.calls[0]["points"][0].id == documents[0].id
    assert client.calls[0]["points"][0].payload["review_id"] == "rvw_a"
    assert client.calls[0]["points"][1].id == documents[1].id
