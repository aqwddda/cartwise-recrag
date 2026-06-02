from __future__ import annotations

from types import SimpleNamespace

import pytest

from cartwise.retrieval.bm25 import BM25Index, BM25Retriever, tokenize


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
        "title": "Portable Microphone Stand",
        "brand": "StageCo",
        "main_category": "Musical Instruments",
        "categories": ["Microphone Accessories"],
        "features": ["Folds for storage"],
        "details_json": "{}",
        "description": "For home recording",
    },
]


def test_tokenize_normalizes_case_and_preserves_model_numbers() -> None:
    assert tokenize("  Fender CD-60S_Guitar ") == ["fender", "cd", "60s", "guitar"]


def test_index_search_returns_only_positive_matches() -> None:
    index = BM25Index.from_items(ITEMS)

    results = index.search("microphone recording", k=10)

    assert [result["parent_asin"] for result in results] == ["P2"]
    assert results[0]["bm25_score"] >= 0
    assert index.search("unmatchedterm", k=10) == []


def test_index_round_trip_preserves_results(tmp_path) -> None:
    index = BM25Index.from_items(ITEMS)
    path = tmp_path / "bm25.json.gz"

    index.save(path)
    loaded = BM25Index.load(path)

    assert loaded.search("guitar tuner", k=2) == index.search("guitar tuner", k=2)


def test_retriever_translates_chinese_before_search() -> None:
    translator = SimpleNamespace(translate=lambda query: "guitar tuner")
    retriever = BM25Retriever(BM25Index.from_items(ITEMS), translator=translator)

    results = retriever.search("吉他调音器")

    assert results[0]["parent_asin"] == "P1"
    assert results[0]["retrieval_query"] == "guitar tuner"
    assert results[0]["retrieval_source"] == "bm25"


def test_index_rejects_duplicate_products() -> None:
    with pytest.raises(ValueError, match="unique"):
        BM25Index(parent_asins=["P1", "P1"], documents=["one", "two"])
