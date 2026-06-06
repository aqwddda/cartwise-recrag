from __future__ import annotations

from typing import Any

from cartwise.application.factory import (
    ApplicationServiceBuildConfig,
    build_application_service,
    evidence_collection_name,
)
from cartwise.core.config import Settings


class FakeApplicationService:
    def __init__(self, *, recommendation_service: Any, evidence_service: Any) -> None:
        self.recommendation_service = recommendation_service
        self.evidence_service = evidence_service


class FakeRecommendationService:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class FakeEvidenceService:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class FakeQdrantClient:
    def __init__(self, collections: list[str]) -> None:
        self.collections = collections

    def get_collection(self, collection: str) -> object:
        self.collections.append(collection)
        return object()


def test_build_application_service_wires_real_service_graph(monkeypatch) -> None:
    collections: list[str] = []
    created: dict[str, Any] = {}

    monkeypatch.setattr(
        "cartwise.application.factory.RecommendationApplicationService",
        FakeApplicationService,
    )
    monkeypatch.setattr(
        "cartwise.application.factory.RecommendationService",
        FakeRecommendationService,
    )
    monkeypatch.setattr(
        "cartwise.application.factory.EvidenceService",
        FakeEvidenceService,
    )
    monkeypatch.setattr(
        "cartwise.application.factory._require_file",
        lambda path, label: path,
    )
    monkeypatch.setattr(
        "cartwise.application.factory.load_items_by_parent_asin",
        lambda path: {"ASIN_1": {"parent_asin": "ASIN_1"}},
    )
    monkeypatch.setattr(
        "cartwise.application.factory.create_qdrant_client",
        lambda url: FakeQdrantClient(collections),
    )
    monkeypatch.setattr(
        "cartwise.application.factory.load_dense_encoder",
        lambda model_key, *, device: {"model_key": model_key, "device": device},
    )
    monkeypatch.setattr(
        "cartwise.application.factory.create_query_translator",
        lambda settings: "translator",
    )
    monkeypatch.setattr(
        "cartwise.application.factory.create_query_intent_parser",
        lambda settings, *, translator: {"translator": translator},
    )
    monkeypatch.setattr(
        "cartwise.application.factory.DenseRetriever",
        lambda client, **kwargs: {"dense": kwargs},
    )
    monkeypatch.setattr(
        "cartwise.application.factory.BM25Retriever",
        lambda index, **kwargs: {"bm25_index": index, **kwargs},
    )
    monkeypatch.setattr(
        "cartwise.application.factory.BM25Index",
        type("FakeBM25Index", (), {"load": staticmethod(lambda path: "bm25-index")}),
    )
    monkeypatch.setattr(
        "cartwise.application.factory.PopularityRecommender",
        type(
            "FakePopularityRecommender",
            (),
            {"from_parquet": staticmethod(lambda path: "popularity")},
        ),
    )
    monkeypatch.setattr(
        "cartwise.application.factory.LightGCNRecommender",
        type(
            "FakeLightGCNRecommender",
            (),
            {"load": staticmethod(lambda path, *, device: {"path": path, "device": device})},
        ),
    )
    monkeypatch.setattr(
        "cartwise.application.factory.QdrantReviewEvidenceRetriever",
        lambda client, **kwargs: {"evidence_retriever": kwargs},
    )
    monkeypatch.setattr(
        "cartwise.application.factory.OpenAICompatibleExplanationGenerator",
        lambda client, *, model: {"client": client, "model": model},
    )
    monkeypatch.setattr(
        "cartwise.application.factory.OpenAI",
        lambda **kwargs: created.setdefault("openai", kwargs),
    )
    monkeypatch.setattr(
        "cartwise.application.factory.httpx.Client",
        lambda **kwargs: created.setdefault("http_client", kwargs),
    )

    service = build_application_service(
        settings=Settings(deepseek_api_key="test-key"),
        config=ApplicationServiceBuildConfig(scope="full", device="cpu"),
    )

    assert isinstance(service, FakeApplicationService)
    assert isinstance(service.recommendation_service, FakeRecommendationService)
    assert isinstance(service.evidence_service, FakeEvidenceService)
    assert collections == [
        "cartwise_products_full_e5_small_v2",
        evidence_collection_name("full", "intfloat/e5-small-v2"),
    ]
    assert service.recommendation_service.kwargs["intent_parser"] == {
        "translator": "translator"
    }
    assert service.recommendation_service.kwargs["dense_retriever"]["dense"][
        "translator"
    ] == "translator"
    assert service.evidence_service.kwargs["generator"]["model"] == "deepseek-v4-flash"
    assert created["http_client"]["trust_env"] is False
