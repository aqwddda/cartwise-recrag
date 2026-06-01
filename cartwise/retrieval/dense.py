"""Dense product indexing and Qdrant retrieval for the stage-six catalog."""

from __future__ import annotations

import json
import math
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import numpy as np
import torch
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from cartwise.core.llm import QueryTranslator, prepare_search_query


PRODUCT_POINT_NAMESPACE = uuid.UUID("cde786cc-a6b4-4a1b-9b8c-411cd8dd7822")


@dataclass(frozen=True, slots=True)
class DenseModelSpec:
    key: str
    model_name: str
    collection_suffix: str


DENSE_MODEL_SPECS = {
    "e5": DenseModelSpec(
        key="e5",
        model_name="intfloat/e5-small-v2",
        collection_suffix="e5_small_v2",
    ),
    "blair": DenseModelSpec(
        key="blair",
        model_name="hyp1231/blair-roberta-base",
        collection_suffix="blair_roberta_base",
    ),
}


@dataclass(frozen=True, slots=True)
class TokenLengthStats:
    documents: int
    minimum: int
    p50: int
    p95: int
    maximum: int
    tokenizer_limit: int
    truncated_documents: int
    truncated_ratio: float


class DenseEncoder(Protocol):
    key: str
    model_name: str
    vector_size: int
    max_sequence_length: int

    def token_lengths(self, documents: Sequence[str]) -> list[int]: ...

    def encode_documents(
        self,
        documents: Sequence[str],
        *,
        batch_size: int,
    ) -> np.ndarray: ...

    def encode_query(self, query: str) -> np.ndarray: ...


def resolve_dense_device(device: str | torch.device) -> torch.device:
    """Resolve an embedding device without silently falling back from CUDA."""

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested, but torch.cuda.is_available() is False")
    return resolved


def _effective_max_length(tokenizer: Any, model_max_length: int | None = None) -> int:
    limit = model_max_length or tokenizer.model_max_length
    if not isinstance(limit, int) or limit <= 0 or limit > 1_000_000:
        raise ValueError("tokenizer does not define a usable maximum sequence length")
    return limit


class E5Encoder:
    """Normalized E5 embeddings with the model card's query/passage prefixes."""

    key = "e5"
    model_name = DENSE_MODEL_SPECS[key].model_name

    def __init__(self, *, device: str | torch.device = "cuda") -> None:
        from sentence_transformers import SentenceTransformer

        self.device = resolve_dense_device(device)
        self.model = SentenceTransformer(self.model_name, device=str(self.device))
        self.max_sequence_length = _effective_max_length(
            self.model.tokenizer,
            self.model.max_seq_length,
        )
        dimension = self.model.get_embedding_dimension()
        if dimension is None:
            raise RuntimeError("E5 model does not expose an embedding dimension")
        self.vector_size = dimension

    def token_lengths(self, documents: Sequence[str]) -> list[int]:
        encoded = self.model.tokenizer(
            [f"passage: {document}" for document in documents],
            truncation=False,
            padding=False,
            verbose=False,
        )
        return [len(token_ids) for token_ids in encoded["input_ids"]]

    def encode_documents(
        self,
        documents: Sequence[str],
        *,
        batch_size: int,
    ) -> np.ndarray:
        return np.asarray(
            self.model.encode(
                [f"passage: {document}" for document in documents],
                batch_size=batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )

    def encode_query(self, query: str) -> np.ndarray:
        return np.asarray(
            self.model.encode(
                f"query: {query}",
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )


class BlairEncoder:
    """Normalized BLaIR CLS embeddings following the model card contract."""

    key = "blair"
    model_name = DENSE_MODEL_SPECS[key].model_name

    def __init__(self, *, device: str | torch.device = "cuda") -> None:
        from transformers import AutoModel, AutoTokenizer

        self.device = resolve_dense_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        self.max_sequence_length = _effective_max_length(self.tokenizer)
        self.vector_size = int(self.model.config.hidden_size)

    def token_lengths(self, documents: Sequence[str]) -> list[int]:
        encoded = self.tokenizer(
            list(documents),
            truncation=False,
            padding=False,
            verbose=False,
        )
        return [len(token_ids) for token_ids in encoded["input_ids"]]

    def _encode(self, texts: Sequence[str], *, batch_size: int) -> np.ndarray:
        batches: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                inputs = self.tokenizer(
                    list(texts[start : start + batch_size]),
                    padding=True,
                    truncation=True,
                    max_length=self.max_sequence_length,
                    return_tensors="pt",
                )
                inputs = {
                    name: value.to(self.device) for name, value in inputs.items()
                }
                cls_embeddings = self.model(**inputs).last_hidden_state[:, 0]
                normalized = torch.nn.functional.normalize(cls_embeddings, p=2, dim=1)
                batches.append(normalized.cpu().numpy())
        if not batches:
            return np.empty((0, self.vector_size), dtype=np.float32)
        return np.concatenate(batches).astype(np.float32, copy=False)

    def encode_documents(
        self,
        documents: Sequence[str],
        *,
        batch_size: int,
    ) -> np.ndarray:
        return self._encode(documents, batch_size=batch_size)

    def encode_query(self, query: str) -> np.ndarray:
        return self._encode([query], batch_size=1)[0]


def load_dense_encoder(
    model_key: str,
    *,
    device: str | torch.device = "cuda",
) -> DenseEncoder:
    if model_key == "e5":
        return E5Encoder(device=device)
    if model_key == "blair":
        return BlairEncoder(device=device)
    raise ValueError(f"unsupported dense model: {model_key}")


def collection_name(scope: str, model_key: str) -> str:
    try:
        suffix = DENSE_MODEL_SPECS[model_key].collection_suffix
    except KeyError as error:
        raise ValueError(f"unsupported dense model: {model_key}") from error
    return f"cartwise_products_{scope}_{suffix}"


def product_point_id(parent_asin: str) -> str:
    return str(uuid.uuid5(PRODUCT_POINT_NAMESPACE, parent_asin))


def _render_sequence(value: Any) -> str:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ""
    return " | ".join(str(entry).strip() for entry in value if str(entry).strip())


def _render_details(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        return ""
    try:
        details = json.loads(value)
    except json.JSONDecodeError:
        return value.strip()
    if not isinstance(details, Mapping):
        return ""
    return " | ".join(
        f"{key}: {json.dumps(detail, ensure_ascii=False, sort_keys=True)}"
        for key, detail in sorted(details.items())
    )


def _render_scalar(value: Any) -> str:
    return "" if value is None else str(value).strip()


def build_product_document(item: Mapping[str, Any]) -> str:
    """Build the shared E5, BLaIR, and later BM25 product document."""

    fields = [
        ("Title", _render_scalar(item.get("title"))),
        ("Brand", _render_scalar(item.get("brand"))),
        ("Main Category", _render_scalar(item.get("main_category"))),
        ("Categories", _render_sequence(item.get("categories"))),
        ("Features", _render_sequence(item.get("features"))),
        ("Details", _render_details(item.get("details_json"))),
        ("Description", _render_scalar(item.get("description"))),
    ]
    return "\n".join(f"{name}: {value}" for name, value in fields)


def summarize_token_lengths(
    lengths: Sequence[int],
    *,
    tokenizer_limit: int,
) -> TokenLengthStats:
    if not lengths:
        raise ValueError("token lengths must not be empty")
    ordered = sorted(lengths)

    def percentile(value: float) -> int:
        return ordered[math.ceil(value * len(ordered)) - 1]

    truncated_documents = sum(length > tokenizer_limit for length in ordered)
    return TokenLengthStats(
        documents=len(ordered),
        minimum=ordered[0],
        p50=percentile(0.50),
        p95=percentile(0.95),
        maximum=ordered[-1],
        tokenizer_limit=tokenizer_limit,
        truncated_documents=truncated_documents,
        truncated_ratio=truncated_documents / len(ordered),
    )


def _json_payload(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_payload(entry) for key, entry in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_payload(entry) for entry in value]
    return str(value)


def build_product_payload(item: Mapping[str, Any], document: str) -> dict[str, Any]:
    return {
        **{str(key): _json_payload(value) for key, value in item.items()},
        "document": document,
    }


def create_qdrant_client(url: str) -> QdrantClient:
    return QdrantClient(url=url, timeout=60, trust_env=False)


def build_qdrant_collection(
    client: Any,
    *,
    collection: str,
    items: Sequence[Mapping[str, Any]],
    encoder: DenseEncoder,
    batch_size: int = 32,
    recreate: bool = False,
) -> dict[str, Any]:
    """Encode a product catalog and write one model-specific Qdrant collection."""

    if not items:
        raise ValueError("items must not be empty")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if client.collection_exists(collection):
        if not recreate:
            raise ValueError(
                f"collection already exists: {collection}; pass --recreate to rebuild it"
            )
        client.delete_collection(collection)

    documents = [build_product_document(item) for item in items]
    token_stats = summarize_token_lengths(
        encoder.token_lengths(documents),
        tokenizer_limit=encoder.max_sequence_length,
    )
    client.create_collection(
        collection_name=collection,
        vectors_config=qdrant_models.VectorParams(
            size=encoder.vector_size,
            distance=qdrant_models.Distance.COSINE,
        ),
    )
    for start in range(0, len(items), batch_size):
        batch_items = items[start : start + batch_size]
        batch_documents = documents[start : start + batch_size]
        vectors = encoder.encode_documents(batch_documents, batch_size=batch_size)
        points = [
            qdrant_models.PointStruct(
                id=product_point_id(str(item["parent_asin"])),
                vector=vector.tolist(),
                payload=build_product_payload(item, document),
            )
            for item, document, vector in zip(
                batch_items,
                batch_documents,
                vectors,
                strict=True,
            )
        ]
        client.upsert(collection_name=collection, points=points, wait=True)
        print(f"Indexed {min(start + len(batch_items), len(items)):,}/{len(items):,}")

    collection_info = client.get_collection(collection)
    return {
        "collection": collection,
        "documents": len(documents),
        "model_key": encoder.key,
        "model_name": encoder.model_name,
        "points_count": collection_info.points_count,
        "token_lengths": asdict(token_stats),
        "vector_size": encoder.vector_size,
    }


class DenseRetriever:
    """Retrieve product payloads from one model-specific Qdrant collection."""

    def __init__(
        self,
        client: Any,
        *,
        collection: str,
        encoder: DenseEncoder,
        translator: QueryTranslator | None = None,
    ) -> None:
        self.client = client
        self.collection = collection
        self.encoder = encoder
        self.translator = translator

    def search(self, query: str, *, k: int = 10) -> list[dict[str, Any]]:
        if k <= 0:
            raise ValueError("k must be greater than zero")
        retrieval_query = prepare_search_query(query, translator=self.translator)
        query_vector = self.encoder.encode_query(retrieval_query)
        response = self.client.query_points(
            collection_name=self.collection,
            query=query_vector.tolist(),
            limit=k,
            with_payload=True,
        )
        results: list[dict[str, Any]] = []
        for point in response.points:
            payload = dict(point.payload or {})
            payload["dense_score"] = point.score
            payload["retrieval_source"] = f"dense:{self.encoder.key}"
            payload["retrieval_query"] = retrieval_query
            results.append(payload)
        return results
