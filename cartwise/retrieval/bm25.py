"""Local BM25 product indexing and retrieval for the stage-six catalog."""

from __future__ import annotations

import gzip
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from cartwise.core.llm import QueryTranslator, prepare_search_query
from cartwise.retrieval.dense import build_product_document


BM25_INDEX_SCHEMA_VERSION = 1
TOKEN_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Tokenize English-oriented retrieval text while preserving model numbers."""

    return TOKEN_PATTERN.findall(text.casefold())


class BM25Index:
    """In-memory BM25 index with a portable local persistence format."""

    def __init__(self, *, parent_asins: Sequence[str], documents: Sequence[str]) -> None:
        if not parent_asins:
            raise ValueError("BM25 index must contain at least one product")
        if len(parent_asins) != len(documents):
            raise ValueError("parent_asins and documents must have the same length")
        if len(set(parent_asins)) != len(parent_asins):
            raise ValueError("BM25 index parent_asins must be unique")
        self.parent_asins = list(parent_asins)
        self.documents = list(documents)
        self._tokenized_documents = [tokenize(document) for document in self.documents]
        self._bm25 = BM25Okapi(self._tokenized_documents)

    @classmethod
    def from_items(cls, items: Sequence[Mapping[str, Any]]) -> BM25Index:
        if not items:
            raise ValueError("items must not be empty")
        parent_asins = [str(item["parent_asin"]).strip() for item in items]
        if any(not parent_asin for parent_asin in parent_asins):
            raise ValueError("items must define non-empty parent_asin values")
        return cls(
            parent_asins=parent_asins,
            documents=[build_product_document(item) for item in items],
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(f"{path.name}.part")
        payload = {
            "schema_version": BM25_INDEX_SCHEMA_VERSION,
            "parent_asins": self.parent_asins,
            "documents": self.documents,
        }
        with gzip.open(partial, "wt", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, separators=(",", ":"))
            file.write("\n")
        partial.replace(path)

    @classmethod
    def load(cls, path: Path) -> BM25Index:
        with gzip.open(path, "rt", encoding="utf-8") as file:
            payload = json.load(file)
        if payload.get("schema_version") != BM25_INDEX_SCHEMA_VERSION:
            raise ValueError(f"unsupported BM25 index schema: {payload.get('schema_version')}")
        return cls(
            parent_asins=payload["parent_asins"],
            documents=payload["documents"],
        )

    def search(self, query: str, *, k: int) -> list[dict[str, Any]]:
        if k <= 0:
            raise ValueError("k must be greater than zero")
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        query_token_set = set(query_tokens)
        scores = self._bm25.get_scores(query_tokens)
        ranked_indexes = np.argsort(-scores, kind="stable")
        return [
            {
                "parent_asin": self.parent_asins[index],
                "document": self.documents[index],
                "bm25_score": float(scores[index]),
            }
            for index in ranked_indexes
            if query_token_set.intersection(self._tokenized_documents[index])
        ][:k]


class BM25Retriever:
    """Retrieve lexical product matches from a local BM25 index."""

    def __init__(
        self,
        index: BM25Index,
        *,
        translator: QueryTranslator | None = None,
    ) -> None:
        self.index = index
        self.translator = translator

    def search(self, query: str, *, k: int = 10) -> list[dict[str, Any]]:
        retrieval_query = prepare_search_query(query, translator=self.translator)
        results = self.index.search(retrieval_query, k=k)
        for result in results:
            result["retrieval_source"] = "bm25"
            result["retrieval_query"] = retrieval_query
        return results
