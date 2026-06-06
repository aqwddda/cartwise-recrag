"""Composition root for the real CartWise recommendation application service."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cartwise.core.config import Settings
from cartwise.evidence.types import (
    DEFAULT_REVIEW_EMBEDDING_MODEL,
    evidence_collection_name,
)
from cartwise.paths import (
    MODELS_ROOT,
    PRODUCT_BM25_ARTIFACT_ROOT,
    PROCESSED_ROOTS,
)

DEFAULT_API_SCOPE = "full"
DEFAULT_API_DEVICE = "cpu"
DEFAULT_DENSE_MODEL_KEY = "e5"


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
) -> Any:
    """Build the real RecommendationApplicationService once for FastAPI startup."""

    active_settings = settings or Settings()
    try:
        return _build_application_service(active_settings, config)
    except ApplicationServiceInitializationError:
        raise
    except ImportError as error:
        dependency = error.name or str(error)
        raise ApplicationServiceInitializationError(
            "failed to initialize RecommendationApplicationService; "
            f"missing runtime dependency: {dependency}"
        ) from error
    except Exception as error:
        raise ApplicationServiceInitializationError(
            f"failed to initialize RecommendationApplicationService: {error}"
        ) from error


def _build_application_service(
    settings: Settings,
    config: ApplicationServiceBuildConfig,
) -> Any:
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
    dense_retriever = create_dense_retriever(
        qdrant_client,
        collection=product_collection,
        encoder=encoder,
        translator=translator,
    )
    bm25_retriever = create_bm25_retriever(
        bm25_path,
        translator=translator,
    )
    popularity_recommender = create_popularity_recommender(training_path)
    lightgcn_recommender = load_lightgcn_recommender(
        lightgcn_path,
        device=config.device,
    )
    recommendation_service = create_recommendation_service(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        popularity_recommender=popularity_recommender,
        lightgcn_recommender=lightgcn_recommender,
        items_by_parent_asin=items_by_parent_asin,
        intent_parser=intent_parser,
    )

    evidence_retriever = create_evidence_retriever(
        qdrant_client,
        collection=evidence_collection,
        encoder=encoder,
    )
    generator = create_explanation_generator(settings)
    evidence_service = create_evidence_service(
        evidence_retriever=evidence_retriever,
        generator=generator,
    )
    return create_recommendation_application_service(
        recommendation_service=recommendation_service,
        evidence_service=evidence_service,
    )


def load_items_by_parent_asin(path: Path) -> dict[str, dict[str, Any]]:
    import pyarrow.parquet as pq

    return {
        item["parent_asin"]: item
        for item in pq.read_table(path).to_pylist()
    }


def create_qdrant_client(qdrant_url: str) -> Any:
    from cartwise.retrieval.dense import create_qdrant_client as build_client

    return build_client(qdrant_url)


def product_collection_name(scope: str, model_key: str) -> str:
    from cartwise.retrieval.dense import collection_name

    return collection_name(scope, model_key)


def load_dense_encoder(model_key: str, *, device: str) -> Any:
    from cartwise.retrieval.dense import load_dense_encoder as load_encoder

    return load_encoder(model_key, device=device)


def create_query_translator(settings: Settings) -> Any:
    from cartwise.query.llm import create_query_translator as build_translator

    return build_translator(settings)


def create_query_intent_parser(settings: Settings, *, translator: Any) -> Any:
    from cartwise.query.llm import create_query_intent_parser as build_parser

    return build_parser(settings, translator=translator)


def create_dense_retriever(
    client: Any,
    *,
    collection: str,
    encoder: Any,
    translator: Any,
) -> Any:
    from cartwise.retrieval.dense import DenseRetriever

    return DenseRetriever(
        client,
        collection=collection,
        encoder=encoder,
        translator=translator,
    )


def create_bm25_retriever(path: Path, *, translator: Any) -> Any:
    from cartwise.retrieval.bm25 import BM25Index, BM25Retriever

    return BM25Retriever(BM25Index.load(path), translator=translator)


def create_popularity_recommender(path: Path) -> Any:
    from cartwise.retrieval.popularity import PopularityRecommender

    return PopularityRecommender.from_parquet(path)


def load_lightgcn_recommender(path: Path, *, device: str) -> Any:
    from cartwise.retrieval.lightgcn import LightGCNRecommender

    return LightGCNRecommender.load(path, device=device)


def create_recommendation_service(
    *,
    dense_retriever: Any,
    bm25_retriever: Any,
    popularity_recommender: Any,
    lightgcn_recommender: Any,
    items_by_parent_asin: dict[str, dict[str, Any]],
    intent_parser: Any,
) -> Any:
    from cartwise.recommendation.service import RecommendationService
    from cartwise.retrieval.filters import resolve_filter_constraints
    from cartwise.retrieval.fusion import FusionConfig

    return RecommendationService(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        popularity_recommender=popularity_recommender,
        lightgcn_recommender=lightgcn_recommender,
        items_by_parent_asin=items_by_parent_asin,
        intent_parser=intent_parser,
        filter_resolver=resolve_filter_constraints,
        fusion_config=FusionConfig(),
    )


def create_evidence_retriever(
    client: Any,
    *,
    collection: str,
    encoder: Any,
) -> Any:
    from cartwise.evidence.rag import QdrantReviewEvidenceRetriever

    return QdrantReviewEvidenceRetriever(
        client,
        collection=collection,
        encoder=encoder,
    )


def create_explanation_generator(settings: Settings) -> Any:
    import httpx
    from openai import OpenAI

    from cartwise.evidence.rag import OpenAICompatibleExplanationGenerator

    return OpenAICompatibleExplanationGenerator(
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


def create_evidence_service(*, evidence_retriever: Any, generator: Any) -> Any:
    from cartwise.evidence.rag import EvidenceRagConfig
    from cartwise.evidence.service import EvidenceService

    return EvidenceService(
        evidence_retriever=evidence_retriever,
        generator=generator,
        config=EvidenceRagConfig(),
    )


def create_recommendation_application_service(
    *,
    recommendation_service: Any,
    evidence_service: Any,
) -> Any:
    from cartwise.application.service import RecommendationApplicationService

    return RecommendationApplicationService(
        recommendation_service=recommendation_service,
        evidence_service=evidence_service,
    )


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
