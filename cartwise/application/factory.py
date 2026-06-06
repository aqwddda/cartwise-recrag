"""Composition root for the real CartWise recommendation application service."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pyarrow.parquet as pq
from openai import OpenAI

from cartwise.application.service import RecommendationApplicationService
from cartwise.core.config import Settings
from cartwise.evidence.rag import (
    EvidenceRagConfig,
    OpenAICompatibleExplanationGenerator,
    QdrantReviewEvidenceRetriever,
)
from cartwise.evidence.service import EvidenceService
from cartwise.paths import (
    MODELS_ROOT,
    PRODUCT_BM25_ARTIFACT_ROOT,
    PROCESSED_ROOTS,
)
from cartwise.query.llm import create_query_intent_parser, create_query_translator
from cartwise.recommendation.service import RecommendationService
from cartwise.retrieval.bm25 import BM25Index, BM25Retriever
from cartwise.retrieval.dense import (
    DenseRetriever,
    collection_name as product_collection_name,
    create_qdrant_client,
    load_dense_encoder,
)
from cartwise.retrieval.filters import resolve_filter_constraints
from cartwise.retrieval.fusion import FusionConfig
from cartwise.retrieval.lightgcn import LightGCNRecommender
from cartwise.retrieval.popularity import PopularityRecommender

DEFAULT_API_SCOPE = "full"
DEFAULT_API_DEVICE = "cuda"
DEFAULT_DENSE_MODEL_KEY = "e5"
DEFAULT_REVIEW_EMBEDDING_MODEL = "intfloat/e5-small-v2"


class ApplicationServiceInitializationError(RuntimeError):
    """Raised when the API composition root cannot build the service graph."""


@dataclass(frozen=True, slots=True)
class ApplicationServiceBuildConfig:
    scope: str = DEFAULT_API_SCOPE
    device: str = DEFAULT_API_DEVICE
    dense_model_key: str = DEFAULT_DENSE_MODEL_KEY
    review_embedding_model: str = DEFAULT_REVIEW_EMBEDDING_MODEL


def build_application_service(
    *,
    settings: Settings | None = None,
    config: ApplicationServiceBuildConfig = ApplicationServiceBuildConfig(),
) -> RecommendationApplicationService:
    """Build the real RecommendationApplicationService once for FastAPI startup."""

    active_settings = settings or Settings()
    try:
        return _build_application_service(active_settings, config)
    except ApplicationServiceInitializationError:
        raise
    except Exception as error:
        raise ApplicationServiceInitializationError(
            f"failed to initialize RecommendationApplicationService: {error}"
        ) from error


def _build_application_service(
    settings: Settings,
    config: ApplicationServiceBuildConfig,
) -> RecommendationApplicationService:
    if config.scope not in PROCESSED_ROOTS:
        raise ApplicationServiceInitializationError(f"unsupported data scope: {config.scope}")
    if config.dense_model_key != DEFAULT_DENSE_MODEL_KEY:
        raise ApplicationServiceInitializationError(
            f"unsupported API dense model: {config.dense_model_key}"
        )
    if settings.llm_api_key is None:
        raise ApplicationServiceInitializationError("LLM API key is not configured")

    processed_root = PROCESSED_ROOTS[config.scope]
    items_path = _require_file(processed_root / "items.parquet", "items parquet")
    training_path = _require_file(
        processed_root / "interactions_train.parquet",
        "training interactions parquet",
    )
    bm25_path = _require_file(
        PRODUCT_BM25_ARTIFACT_ROOT / config.scope / "bm25.json.gz",
        "BM25 index",
    )
    lightgcn_path = _require_file(
        MODELS_ROOT / "lightgcn" / config.scope / "lightgcn.pt",
        "LightGCN model",
    )

    items_by_parent_asin = load_items_by_parent_asin(items_path)
    qdrant_client = create_qdrant_client(settings.qdrant_url)
    product_collection = product_collection_name(config.scope, config.dense_model_key)
    evidence_collection = evidence_collection_name(
        config.scope,
        config.review_embedding_model,
    )
    _require_qdrant_collection(qdrant_client, product_collection, "product dense")
    _require_qdrant_collection(qdrant_client, evidence_collection, "review evidence")

    encoder = load_dense_encoder(config.dense_model_key, device=config.device)
    translator = create_query_translator(settings)
    intent_parser = create_query_intent_parser(settings, translator=translator)
    dense_retriever = DenseRetriever(
        qdrant_client,
        collection=product_collection,
        encoder=encoder,
        translator=translator,
    )
    bm25_retriever = BM25Retriever(
        BM25Index.load(bm25_path),
        translator=translator,
    )
    popularity_recommender = PopularityRecommender.from_parquet(training_path)
    lightgcn_recommender = LightGCNRecommender.load(
        lightgcn_path,
        device=config.device,
    )
    recommendation_service = RecommendationService(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        popularity_recommender=popularity_recommender,
        lightgcn_recommender=lightgcn_recommender,
        items_by_parent_asin=items_by_parent_asin,
        intent_parser=intent_parser,
        filter_resolver=resolve_filter_constraints,
        fusion_config=FusionConfig(),
    )

    evidence_retriever = QdrantReviewEvidenceRetriever(
        qdrant_client,
        collection=evidence_collection,
        encoder=encoder,
    )
    generator = OpenAICompatibleExplanationGenerator(
        OpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            timeout=settings.llm_timeout_seconds,
            http_client=httpx.Client(
                proxy=settings.external_https_proxy,
                trust_env=False,
            ),
        ),
        model=settings.llm_model,
    )
    evidence_service = EvidenceService(
        evidence_retriever=evidence_retriever,
        generator=generator,
        config=EvidenceRagConfig(),
    )
    return RecommendationApplicationService(
        recommendation_service=recommendation_service,
        evidence_service=evidence_service,
    )


def load_items_by_parent_asin(path: Path) -> dict[str, dict[str, Any]]:
    return {
        item["parent_asin"]: item
        for item in pq.read_table(path).to_pylist()
    }


def evidence_collection_name(scope: str, model_name: str) -> str:
    return f"cartwise_review_evidence_{scope}_{_model_slug(model_name)}"


def _model_slug(model_name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", model_name).strip("_").lower()
    if not slug:
        raise ValueError("model name must contain at least one alphanumeric character")
    return slug


def _require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _require_qdrant_collection(client: Any, collection: str, label: str) -> None:
    try:
        client.get_collection(collection)
    except Exception as error:
        raise ApplicationServiceInitializationError(
            f"{label} Qdrant collection is unavailable: {collection}"
        ) from error
