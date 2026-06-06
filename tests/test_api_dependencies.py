from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from cartwise.api.main import create_app
from cartwise.application.factory import (
    ApplicationServiceBuildConfig,
    DEFAULT_API_DEVICE,
    build_application_service,
    evidence_collection_name,
    product_collection_name,
)
from cartwise.core.config import Settings
from cartwise.evidence.collections import evidence_collection_name as shared_evidence_collection_name
from cartwise.retrieval.collection_names import product_collection_name as shared_product_collection_name
from tests.test_api import FakeApplicationService


class FakeQdrantClient:
    def __init__(self, collections: list[str]) -> None:
        self.collections = collections

    def get_collection(self, collection: str) -> object:
        self.collections.append(collection)
        return object()


def test_build_application_service_wires_real_service_graph(monkeypatch) -> None:
    service_graph: dict[str, Any] = {}
    collections: list[str] = []
    created: dict[str, Any] = {}

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
        "cartwise.application.factory.create_dense_retriever",
        lambda client, **kwargs: {"dense": kwargs},
    )
    monkeypatch.setattr(
        "cartwise.application.factory.create_bm25_retriever",
        lambda path, **kwargs: {"bm25_path": path, **kwargs},
    )
    monkeypatch.setattr(
        "cartwise.application.factory.create_popularity_recommender",
        lambda path: "popularity",
    )
    monkeypatch.setattr(
        "cartwise.application.factory.load_lightgcn_recommender",
        lambda path, *, device: {"path": path, "device": device},
    )

    def fake_recommendation_service(**kwargs: Any) -> dict[str, Any]:
        service_graph["recommendation"] = kwargs
        return {"recommendation_service": kwargs}

    monkeypatch.setattr(
        "cartwise.application.factory.create_recommendation_service",
        fake_recommendation_service,
    )
    monkeypatch.setattr(
        "cartwise.application.factory.create_evidence_retriever",
        lambda client, **kwargs: {"evidence_retriever": kwargs},
    )
    monkeypatch.setattr(
        "cartwise.application.factory.create_explanation_generator",
        lambda settings: created.setdefault("generator", {"model": settings.llm_model}),
    )

    def fake_evidence_service(**kwargs: Any) -> dict[str, Any]:
        service_graph["evidence"] = kwargs
        return {"evidence_service": kwargs}

    monkeypatch.setattr(
        "cartwise.application.factory.create_evidence_service",
        fake_evidence_service,
    )

    def fake_application_service(**kwargs: Any) -> dict[str, Any]:
        service_graph["application"] = kwargs
        return {"application_service": kwargs}

    monkeypatch.setattr(
        "cartwise.application.factory.create_recommendation_application_service",
        fake_application_service,
    )

    service = build_application_service(
        settings=Settings(deepseek_api_key="test-key"),
        config=ApplicationServiceBuildConfig(scope="full", device="cpu"),
    )

    assert service == {"application_service": service_graph["application"]}
    assert collections == [
        "cartwise_products_full_e5_small_v2",
        evidence_collection_name("full", "intfloat/e5-small-v2"),
    ]
    assert service_graph["recommendation"]["intent_parser"] == {
        "translator": "translator"
    }
    assert service_graph["recommendation"]["dense_retriever"]["dense"][
        "translator"
    ] == "translator"
    assert service_graph["evidence"]["generator"]["model"] == "deepseek-v4-flash"


def test_default_build_config_uses_cpu_and_cuda_can_be_requested() -> None:
    assert DEFAULT_API_DEVICE == "cpu"
    assert ApplicationServiceBuildConfig().device == "cpu"
    assert ApplicationServiceBuildConfig(device="cuda").device == "cuda"


def test_builder_passes_explicit_cuda_device_to_model_loaders(monkeypatch) -> None:
    devices: dict[str, str] = {}

    monkeypatch.setattr(
        "cartwise.application.factory._require_file",
        lambda path, label: path,
    )
    monkeypatch.setattr(
        "cartwise.application.factory.load_items_by_parent_asin",
        lambda path: {},
    )
    monkeypatch.setattr(
        "cartwise.application.factory.create_qdrant_client",
        lambda url: FakeQdrantClient([]),
    )
    monkeypatch.setattr(
        "cartwise.application.factory.load_dense_encoder",
        lambda model_key, *, device: devices.setdefault("dense", device),
    )
    monkeypatch.setattr(
        "cartwise.application.factory.create_query_translator",
        lambda settings: "translator",
    )
    monkeypatch.setattr(
        "cartwise.application.factory.create_query_intent_parser",
        lambda settings, *, translator: "intent-parser",
    )
    monkeypatch.setattr(
        "cartwise.application.factory.create_dense_retriever",
        lambda client, **kwargs: "dense-retriever",
    )
    monkeypatch.setattr(
        "cartwise.application.factory.create_bm25_retriever",
        lambda path, **kwargs: "bm25-retriever",
    )
    monkeypatch.setattr(
        "cartwise.application.factory.create_popularity_recommender",
        lambda path: "popularity",
    )
    monkeypatch.setattr(
        "cartwise.application.factory.load_lightgcn_recommender",
        lambda path, *, device: devices.setdefault("lightgcn", device),
    )
    monkeypatch.setattr(
        "cartwise.application.factory.create_recommendation_service",
        lambda **kwargs: "recommendation-service",
    )
    monkeypatch.setattr(
        "cartwise.application.factory.create_evidence_retriever",
        lambda client, **kwargs: "evidence-retriever",
    )
    monkeypatch.setattr(
        "cartwise.application.factory.create_explanation_generator",
        lambda settings: "generator",
    )
    monkeypatch.setattr(
        "cartwise.application.factory.create_evidence_service",
        lambda **kwargs: "evidence-service",
    )
    monkeypatch.setattr(
        "cartwise.application.factory.create_recommendation_application_service",
        lambda **kwargs: "application-service",
    )

    service = build_application_service(
        settings=Settings(deepseek_api_key="test-key"),
        config=ApplicationServiceBuildConfig(device="cuda"),
    )

    assert service == "application-service"
    assert devices == {"dense": "cuda", "lightgcn": "cuda"}


def test_evidence_collection_name_reuses_shared_indexing_rule() -> None:
    assert evidence_collection_name("dev", "intfloat/e5-small-v2") == (
        shared_evidence_collection_name("dev", "intfloat/e5-small-v2")
    )
    assert evidence_collection_name("dev", "intfloat/e5-small-v2") == (
        "cartwise_review_evidence_dev_intfloat_e5_small_v2"
    )


def test_product_collection_name_uses_lightweight_naming_rule() -> None:
    assert product_collection_name("full", "e5") == shared_product_collection_name(
        "full",
        "e5",
    )
    assert product_collection_name("full", "e5") == (
        "cartwise_products_full_e5_small_v2"
    )


def test_fake_service_app_does_not_call_real_builder(monkeypatch) -> None:
    calls = 0

    def forbidden_builder():
        nonlocal calls
        calls += 1
        raise AssertionError("real builder should not run for fake-service app")

    monkeypatch.setattr(
        "cartwise.api.main.build_application_service",
        forbidden_builder,
    )
    app = create_app(application_service=FakeApplicationService())

    with TestClient(app) as client:
        response = client.get("/health/ready")

    assert response.status_code == 200
    assert calls == 0
