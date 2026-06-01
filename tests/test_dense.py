from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from cartwise.retrieval.dense import (
    DenseRetriever,
    build_product_document,
    build_qdrant_collection,
    collection_name,
    product_point_id,
    summarize_token_lengths,
)


class FakeEncoder:
    key = "fake"
    model_name = "fake/model"
    vector_size = 2
    max_sequence_length = 5

    def token_lengths(self, documents):
        return [len(document.split()) for document in documents]

    def encode_documents(self, documents, *, batch_size):
        return np.asarray([[1.0, 0.0] for _ in documents], dtype=np.float32)

    def encode_query(self, query):
        return np.asarray([0.0, 1.0], dtype=np.float32)


class FakeQdrantClient:
    def __init__(self, *, exists: bool = False) -> None:
        self.exists = exists
        self.deleted: list[str] = []
        self.created: list[tuple[str, object]] = []
        self.upserted: list[object] = []
        self.queries: list[dict[str, object]] = []

    def collection_exists(self, collection):
        return self.exists

    def delete_collection(self, collection):
        self.deleted.append(collection)
        self.exists = False

    def create_collection(self, *, collection_name, vectors_config):
        self.created.append((collection_name, vectors_config))
        self.exists = True

    def upsert(self, *, collection_name, points, wait):
        self.upserted.extend(points)

    def get_collection(self, collection):
        return SimpleNamespace(points_count=len(self.upserted))

    def query_points(self, **kwargs):
        self.queries.append(kwargs)
        return SimpleNamespace(
            points=[
                SimpleNamespace(
                    score=0.9,
                    payload={"parent_asin": "P1", "title": "Guitar Tuner"},
                )
            ]
        )


ITEMS = [
    {
        "parent_asin": "P1",
        "title": "Clip-On Guitar Tuner",
        "brand": "Example",
        "main_category": "Musical Instruments",
        "categories": ["Instrument Accessories", "Tuners"],
        "features": ["Compact"],
        "details_json": '{"Color": "Black"}',
        "description": "Useful for beginners",
    },
    {
        "parent_asin": "P2",
        "title": "Microphone Stand",
        "brand": None,
        "main_category": "Musical Instruments",
        "categories": [],
        "features": [],
        "details_json": "{}",
        "description": None,
    },
]


def test_product_document_uses_fixed_high_value_field_order() -> None:
    document = build_product_document(ITEMS[0])

    assert [line.split(":", 1)[0] for line in document.splitlines()] == [
        "Title",
        "Brand",
        "Main Category",
        "Categories",
        "Features",
        "Details",
        "Description",
    ]
    assert "Title: Clip-On Guitar Tuner" in document
    assert 'Details: Color: "Black"' in document


def test_token_stats_record_tokenizer_truncation() -> None:
    stats = summarize_token_lengths([1, 3, 6, 10], tokenizer_limit=5)

    assert stats.p50 == 3
    assert stats.p95 == 10
    assert stats.truncated_documents == 2
    assert stats.truncated_ratio == 0.5


def test_product_point_id_is_stable_and_collection_names_are_model_specific() -> None:
    assert product_point_id("P1") == product_point_id("P1")
    assert product_point_id("P1") != product_point_id("P2")
    assert collection_name("dev", "e5") != collection_name("dev", "blair")


def test_build_collection_writes_payloads_and_token_stats() -> None:
    client = FakeQdrantClient()

    report = build_qdrant_collection(
        client,
        collection="products",
        items=ITEMS,
        encoder=FakeEncoder(),
        batch_size=1,
    )

    assert report["points_count"] == 2
    assert report["token_lengths"]["documents"] == 2
    assert client.created[0][0] == "products"
    assert [point.payload["parent_asin"] for point in client.upserted] == ["P1", "P2"]
    assert all(point.payload["document"] for point in client.upserted)


def test_build_collection_refuses_implicit_overwrite() -> None:
    client = FakeQdrantClient(exists=True)

    with pytest.raises(ValueError, match="--recreate"):
        build_qdrant_collection(
            client,
            collection="products",
            items=ITEMS,
            encoder=FakeEncoder(),
        )


def test_dense_retriever_translates_chinese_before_vector_search() -> None:
    client = FakeQdrantClient()
    translator = SimpleNamespace(translate=lambda query: "guitar tuner")
    retriever = DenseRetriever(
        client,
        collection="products",
        encoder=FakeEncoder(),
        translator=translator,
    )

    results = retriever.search("吉他调音器", k=1)

    assert client.queries[0]["query"] == [0.0, 1.0]
    assert results[0]["retrieval_query"] == "guitar tuner"
    assert results[0]["retrieval_source"] == "dense:fake"
