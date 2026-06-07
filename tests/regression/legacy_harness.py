from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from cartwise.evidence.rag import explain_candidates
from cartwise.query.llm import ParsedQueryIntent
from cartwise.retrieval.filters import FilterConstraints
from cartwise.retrieval.fusion import FusionConfig
from scripts.tools import audit_retrieval


ITEMS_BY_PARENT_ASIN: dict[str, dict[str, Any]] = {
    "TUNER_A": {
        "parent_asin": "TUNER_A",
        "title": "Clip-On Guitar Tuner",
        "brand": "Snark",
        "price": 19.99,
        "categories": ["Musical Instruments", "Guitar Tuners"],
        "details_json": json.dumps({"Color": "black"}),
    },
    "TUNER_B": {
        "parent_asin": "TUNER_B",
        "title": "Pedal Guitar Tuner",
        "brand": "Boss",
        "price": 79.99,
        "categories": ["Musical Instruments", "Guitar Tuners"],
        "details_json": json.dumps({"Color": "white"}),
    },
    "MIC_A": {
        "parent_asin": "MIC_A",
        "title": "Studio Vocal Microphone",
        "brand": "Shure",
        "price": 99.0,
        "categories": ["Musical Instruments", "Microphones"],
        "details_json": json.dumps({"Material": "metal"}),
    },
    "STAND_A": {
        "parent_asin": "STAND_A",
        "title": "Portable Microphone Stand",
        "brand": "OnStage",
        "price": 39.0,
        "categories": ["Musical Instruments", "Microphone Stands"],
        "details_json": json.dumps({"Material": "steel"}),
    },
    "FENDER_A": {
        "parent_asin": "FENDER_A",
        "title": "Fender Guitar Tuner",
        "brand": "Fender",
        "price": 24.0,
        "categories": ["Musical Instruments", "Guitar Tuners"],
        "details_json": "{}",
    },
    "OVER_BUDGET": {
        "parent_asin": "OVER_BUDGET",
        "title": "Premium Guitar Tuner",
        "brand": "Boss",
        "price": 120.0,
        "categories": ["Musical Instruments", "Guitar Tuners"],
        "details_json": "{}",
    },
}


class FakeIntentParser:
    def parse(self, query: str) -> ParsedQueryIntent:
        normalized = query.casefold()
        if "empty" in normalized:
            return ParsedQueryIntent(
                search_query=query,
                product_terms=("empty",),
                filters=FilterConstraints(),
            )
        if "适合" in query:
            return ParsedQueryIntent(
                search_query="beginner guitar tuner",
                product_terms=("guitar tuner",),
                filters=FilterConstraints(),
            )
        if "under 50" in normalized:
            return ParsedQueryIntent(
                search_query=query,
                product_terms=("guitar tuner",),
                filters=FilterConstraints(max_price=50),
            )
        if "not usb" in normalized:
            return ParsedQueryIntent(
                search_query=query,
                product_terms=("microphone",),
                filters=FilterConstraints(excluded_brands=("USB",)),
            )
        return ParsedQueryIntent(
            search_query=query,
            product_terms=("guitar tuner",),
            filters=FilterConstraints(),
        )


class FakeDenseRetriever:
    def search(self, query: str, *, k: int) -> list[dict[str, Any]]:
        if "empty" in query.casefold():
            return []
        if "microphone" in query.casefold():
            rows = [("MIC_A", 0.91), ("STAND_A", 0.72)]
        else:
            rows = [("TUNER_A", 0.95), ("TUNER_B", 0.83), ("FENDER_A", 0.77)]
        return [
            {
                "parent_asin": parent_asin,
                "dense_score": score,
                "retrieval_query": query,
                "document": f"dense document {parent_asin}",
            }
            for parent_asin, score in rows[:k]
        ]


class FakeBM25Retriever:
    def search(self, query: str, *, k: int) -> list[dict[str, Any]]:
        if "empty" in query.casefold():
            return []
        if "microphone" in query.casefold():
            rows = [("STAND_A", 3.5), ("MIC_A", 2.5)]
        else:
            rows = [("TUNER_B", 4.0), ("TUNER_A", 2.0), ("OVER_BUDGET", 1.0)]
        return [
            {
                "parent_asin": parent_asin,
                "bm25_score": score,
                "retrieval_query": query,
                "document": f"bm25 document {parent_asin}",
            }
            for parent_asin, score in rows[:k]
        ]


class FakeLightGCNRecommender:
    user_to_index = {"known-user": 0}

    def recommend(self, user_id: str, *, k: int) -> list[str]:
        if user_id not in self.user_to_index:
            return []
        return ["TUNER_B", "FENDER_A", "TUNER_A"][:k]


class FakePopularityRecommender:
    item_counts = {
        "TUNER_A": 30,
        "TUNER_B": 20,
        "STAND_A": 10,
        "MIC_A": 8,
        "FENDER_A": 4,
        "OVER_BUDGET": 2,
    }

    def recommend(self, user_id: str, *, k: int) -> list[str]:
        del user_id
        return ["TUNER_A", "STAND_A", "MIC_A", "OVER_BUDGET"][:k]


class FakeEvidenceRetriever:
    def __init__(self, *, no_evidence: bool = False) -> None:
        self.no_evidence = no_evidence

    def search(
        self,
        query: str,
        *,
        parent_asin: str,
        k: int,
        rating_lte: float | None = None,
    ) -> list[Mapping[str, Any]]:
        del query, k
        if self.no_evidence:
            return []
        if rating_lte is not None:
            return [
                {
                    "review_id": f"{parent_asin}-LOW",
                    "chunk_id": f"{parent_asin}-LOW-0",
                    "parent_asin": parent_asin,
                    "rating": 2.0,
                    "title": "Concern",
                    "text": "Some users mention setup issues.",
                    "chunk_text": "Some users mention setup issues.",
                    "score": 0.55,
                }
            ]
        return [
            {
                "review_id": f"{parent_asin}-R1",
                "chunk_id": f"{parent_asin}-R1-0",
                "parent_asin": parent_asin,
                "rating": 5.0,
                "title": "Great",
                "text": "Works well for practice.",
                "chunk_text": "Works well for practice.",
                "score": 0.9,
            },
            {
                "review_id": f"{parent_asin}-R2",
                "chunk_id": f"{parent_asin}-R2-0",
                "parent_asin": parent_asin,
                "rating": 4.0,
                "title": "Good",
                "text": "Good value.",
                "chunk_text": "Good value.",
                "score": 0.8,
            },
        ]


def _constraints_payload(constraints: FilterConstraints) -> dict[str, Any]:
    return {
        "category_tags": list(constraints.category_tags),
        "min_price": constraints.min_price,
        "max_price": constraints.max_price,
        "brands": list(constraints.brands),
        "excluded_brands": list(constraints.excluded_brands),
        "color_tags": list(constraints.color_tags),
        "material_tags": list(constraints.material_tags),
    }


def _patched_resolver(**kwargs: Any) -> FilterConstraints:
    product_terms = tuple(kwargs.get("product_terms", ()))
    category_tags = ()
    if product_terms == ("guitar tuner",):
        category_tags = ("Guitar Tuners",)
    elif product_terms == ("microphone",):
        category_tags = ("Microphones",)
    elif product_terms == ("empty",):
        category_tags = ("No Matching Category",)
    return FilterConstraints(
        category_tags=category_tags,
        min_price=kwargs.get("min_price"),
        max_price=kwargs.get("max_price"),
        brands=kwargs.get("brands", ()),
        excluded_brands=kwargs.get("excluded_brands", ()),
        color_tags=kwargs.get("color_tags", ()),
        material_tags=kwargs.get("material_tags", ()),
    )


def run_legacy_fusion_cases() -> dict[str, Any]:
    original_resolver = audit_retrieval.resolve_filter_constraints
    audit_retrieval.resolve_filter_constraints = _patched_resolver
    try:
        channel = audit_retrieval.FusionAuditChannel(
            dense_retriever=FakeDenseRetriever(),
            bm25_retriever=FakeBM25Retriever(),
            lightgcn_recommender=FakeLightGCNRecommender(),
            popularity_recommender=FakePopularityRecommender(),
            items_by_parent_asin=ITEMS_BY_PARENT_ASIN,
            intent_parser=FakeIntentParser(),
            config=FusionConfig(dense_k=3, bm25_k=3, lightgcn_k=3, popularity_k=4, final_top_k=5),
        )
        cases = [
            ("english", "guitar tuner for beginners", None),
            ("chinese", "适合初学者的吉他调音器", None),
            ("price", "guitar tuner under 50 dollars", None),
            ("negative", "microphone for recording vocals but not USB", None),
            ("known_user", "guitar tuner for beginners", "known-user"),
            ("unknown_user", "guitar tuner for beginners", "unknown-user"),
            ("no_user", "guitar tuner for beginners", None),
            ("empty", "empty candidate query", None),
        ]
        results: dict[str, Any] = {}
        for name, query, user_id in cases:
            final_results = channel.recall(query, k=5, user_id=user_id)
            results[name] = {
                "query": query,
                "user_id": user_id,
                "known_user": bool(user_id and user_id in FakeLightGCNRecommender.user_to_index),
                "fusion_intent": channel.audit_metadata()["fusion_intent"],
                "filter_constraints": channel.audit_metadata()["filter_constraints"],
                "final_parent_asins": [result["parent_asin"] for result in final_results],
                "final_results": [
                    {
                        "parent_asin": result["parent_asin"],
                        "rank": result["rank"],
                        "fusion_score": result["fusion_score"],
                        "sources": result["sources"],
                        "source_ranks": result["source_ranks"],
                        "source_scores": result["source_scores"],
                    }
                    for result in final_results
                ],
                "filtered_results": [
                    {
                        "parent_asin": result["parent_asin"],
                        "sources": result["sources"],
                        "filter_policy": result["filter_policy"],
                        "filter_reason": result["filter_reason"],
                    }
                    for result in channel._last_filtered_results
                ],
            }
        return results
    finally:
        audit_retrieval.resolve_filter_constraints = original_resolver


def run_legacy_evidence_cases() -> dict[str, Any]:
    candidate = {
        "parent_asin": "TUNER_A",
        "item": ITEMS_BY_PARENT_ASIN["TUNER_A"],
    }
    with_evidence = explain_candidates(
        english_query="guitar tuner for beginners",
        candidates=[candidate],
        retriever=FakeEvidenceRetriever(),
        generator=None,
    )[0]
    no_evidence = explain_candidates(
        english_query="guitar tuner for beginners",
        candidates=[candidate],
        retriever=FakeEvidenceRetriever(no_evidence=True),
        generator=None,
    )[0]
    return {
        "with_evidence": {
            "parent_asin": with_evidence.parent_asin,
            "fallback": with_evidence.fallback,
            "cited_review_ids": list(with_evidence.cited_review_ids),
            "evidence_review_ids": [entry.review_id for entry in with_evidence.evidence],
            "evidence_ratings": [entry.rating for entry in with_evidence.evidence],
            "has_reason": bool(with_evidence.reason),
            "has_potential_cons": bool(with_evidence.potential_cons),
        },
        "no_evidence": {
            "parent_asin": no_evidence.parent_asin,
            "fallback": no_evidence.fallback,
            "cited_review_ids": list(no_evidence.cited_review_ids),
            "evidence_review_ids": [entry.review_id for entry in no_evidence.evidence],
            "has_reason": bool(no_evidence.reason),
            "has_potential_cons": bool(no_evidence.potential_cons),
        },
    }
