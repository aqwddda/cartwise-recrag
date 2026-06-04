"""Build the stage-eight LangChain dense review evidence Qdrant index."""

from __future__ import annotations

import argparse
import json
import math
import re
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from transformers import AutoTokenizer

from cartwise.core.config import Settings
from scripts.paths import EVIDENCE_DENSE_ARTIFACT_ROOT, PROCESSED_ROOTS


DEFAULT_REVIEW_EMBEDDING_MODEL = "intfloat/e5-small-v2"
DEFAULT_CHUNK_SIZE = 384
DEFAULT_CHUNK_OVERLAP = 64
DEFAULT_BATCH_SIZE = 512
DEFAULT_REVIEW_BATCH_SIZE = 5_000
REVIEW_POINT_NAMESPACE = uuid.UUID("d905ef4d-b156-43e9-a02c-00e439a1f17d")


@dataclass(frozen=True, slots=True)
class TokenLengthStats:
    documents: int
    minimum: int
    p50: int
    p90: int
    p95: int
    p99: int
    maximum: int


@dataclass(frozen=True, slots=True)
class BuildProgress:
    batches_processed: int = 0
    reviews_seen: int = 0
    chunks_indexed: int = 0
    split_reviews: int = 0
    skipped_reviews: int = 0
    parent_asin_count: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=PROCESSED_ROOTS, default="dev")
    parser.add_argument("--processed-root", type=Path)
    parser.add_argument("--qdrant-url")
    parser.add_argument("--collection")
    parser.add_argument("--model-name", default=DEFAULT_REVIEW_EMBEDDING_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--review-batch-size", type=int, default=DEFAULT_REVIEW_BATCH_SIZE)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("--checkpoint-output", type=Path)
    return parser.parse_args()


def resolve_device(device: str | torch.device) -> torch.device:
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested, but torch.cuda.is_available() is False")
    return resolved


def model_slug(model_name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", model_name).strip("_").lower()
    if not slug:
        raise ValueError("model name must contain at least one alphanumeric character")
    return slug


def collection_name(scope: str, model_name: str) -> str:
    return f"cartwise_review_evidence_{scope}_{model_slug(model_name)}"


def review_point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(REVIEW_POINT_NAMESPACE, chunk_id))


def load_reviews(processed_root: Path) -> list[dict[str, Any]]:
    path = processed_root / "reviews.parquet"
    if not path.exists():
        raise FileNotFoundError(f"reviews parquet not found: {path}")
    return pq.read_table(path).to_pylist()


def iter_review_batches(
    processed_root: Path,
    *,
    batch_size: int,
) -> tuple[Path, int, Any]:
    if batch_size <= 0:
        raise ValueError("review_batch_size must be greater than zero")
    path = processed_root / "reviews.parquet"
    if not path.exists():
        raise FileNotFoundError(f"reviews parquet not found: {path}")
    parquet_file = pq.ParquetFile(path)
    return path, parquet_file.metadata.num_rows, parquet_file.iter_batches(
        batch_size=batch_size
    )


def build_review_text(review: Mapping[str, Any]) -> str:
    fields = [
        ("Review title", review.get("title")),
        ("Rating", review.get("rating")),
        ("Review text", review.get("text")),
    ]
    return "\n".join(
        f"{label}: {value}"
        for label, value in fields
        if value is not None and str(value).strip()
    )


def create_token_length_function(model_name: str) -> Callable[[str], int]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def token_length(text: str) -> int:
        encoded = tokenizer(
            f"passage: {text}",
            truncation=False,
            padding=False,
            verbose=False,
        )
        return len(encoded["input_ids"])

    return token_length


def create_text_splitter(
    *,
    chunk_size: int,
    chunk_overlap: int,
    length_function: Callable[[str], int],
) -> RecursiveCharacterTextSplitter:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=length_function,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
    )


def build_review_documents(
    reviews: Sequence[Mapping[str, Any]],
    *,
    split_text: Callable[[str], list[str]],
) -> tuple[list[Document], dict[str, int]]:
    documents: list[Document] = []
    split_reviews = 0
    skipped_reviews = 0
    for review in reviews:
        review_id = str(review.get("review_id") or "").strip()
        parent_asin = str(review.get("parent_asin") or "").strip()
        if not review_id or not parent_asin:
            skipped_reviews += 1
            continue
        review_text = build_review_text(review)
        if not review_text.strip():
            skipped_reviews += 1
            continue
        text_chunks = split_text(review_text) or [review_text]
        if len(text_chunks) > 1:
            split_reviews += 1
        for chunk_index, chunk_text in enumerate(text_chunks):
            chunk_id = f"{review_id}#chunk_{chunk_index}"
            metadata = {
                "chunk_id": chunk_id,
                "review_id": review_id,
                "parent_asin": parent_asin,
                "rating": review.get("rating"),
                "title": review.get("title"),
                "text": review.get("text"),
                "chunk_text": chunk_text,
                "helpful_vote": review.get("helpful_vote"),
                "verified_purchase": review.get("verified_purchase"),
                "timestamp": review.get("timestamp"),
            }
            documents.append(
                Document(
                    id=review_point_id(chunk_id),
                    page_content=f"passage: {chunk_text}",
                    metadata=metadata,
                )
            )
    return documents, {
        "split_reviews": split_reviews,
        "skipped_reviews": skipped_reviews,
    }


def summarize_token_lengths(lengths: Sequence[int]) -> TokenLengthStats:
    if not lengths:
        raise ValueError("token lengths must not be empty")
    ordered = sorted(lengths)

    def percentile(value: float) -> int:
        return ordered[math.ceil(value * len(ordered)) - 1]

    return TokenLengthStats(
        documents=len(ordered),
        minimum=ordered[0],
        p50=percentile(0.50),
        p90=percentile(0.90),
        p95=percentile(0.95),
        p99=percentile(0.99),
        maximum=ordered[-1],
    )


def create_embeddings(
    *,
    model_name: str,
    device: str,
    batch_size: int,
) -> SentenceTransformer:
    resolved_device = resolve_device(device)
    model = SentenceTransformer(model_name, device=str(resolved_device))
    model.half()  # FP16 — roughly 2x throughput on Turing+
    print(f"Embedding model loaded: {model_name} (FP16, device={resolved_device})", flush=True)
    return model


def create_qdrant_client(qdrant_url: str) -> QdrantClient:
    return QdrantClient(url=qdrant_url, timeout=60, trust_env=False)


def ensure_qdrant_collection(
    *,
    collection: str,
    embeddings: SentenceTransformer,
    client: QdrantClient,
    recreate: bool,
) -> None:
    if client.collection_exists(collection):
        if not recreate:
            return
        client.delete_collection(collection)
    dimension = embeddings.get_embedding_dimension()  # was get_sentence_embedding_dimension
    client.create_collection(
        collection_name=collection,
        vectors_config=qdrant_models.VectorParams(
            size=dimension,
            distance=qdrant_models.Distance.COSINE,
        ),
        optimizers_config=qdrant_models.OptimizersConfigDiff(
            indexing_threshold=20_000,
        ),
    )


def qdrant_payload(document: Document) -> dict[str, Any]:
    metadata = dict(document.metadata)
    return {
        "parent_asin": metadata["parent_asin"],
        "review_id": metadata["review_id"],
        "chunk_id": metadata["chunk_id"],
        "rating": metadata["rating"],
        "chunk_text": metadata["chunk_text"],
    }


def upsert_documents_to_qdrant(
    *,
    client: QdrantClient,
    collection: str,
    embeddings: SentenceTransformer,
    documents: Sequence[Document],
    batch_size: int,
    progress_callback: Callable[[int], None] | None = None,
) -> int:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    added = 0
    encode_batch = batch_size  # GPU batch — may shrink on OOM
    outer_step = batch_size * 2  # accumulate more docs per round to reduce loop overhead
    cursor = 0
    total = len(documents)
    while cursor < total:
        end = min(cursor + outer_step, total)
        batch = list(documents[cursor:end])
        texts = [document.page_content for document in batch]
        while True:
            try:
                vectors = embeddings.encode(
                    texts,
                    batch_size=encode_batch,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                break
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower() and encode_batch > 64:
                    encode_batch = max(64, encode_batch // 2)
                    print(
                        f"OOM at encode_batch={encode_batch * 2}; "
                        f"falling back to encode_batch={encode_batch} "
                        f"(outer step={outer_step})",
                        flush=True,
                    )
                    torch.cuda.empty_cache()
                    continue
                raise
        points = [
            qdrant_models.PointStruct(
                id=document.id,
                vector=vector.tolist(),
                payload=qdrant_payload(document),
            )
            for document, vector in zip(batch, vectors, strict=True)
        ]
        client.upsert(
            collection_name=collection,
            points=points,
            wait=False,
        )
        added += len(batch)
        cursor = end
        if progress_callback is not None:
            progress_callback(added)
    return added


def add_documents_to_vector_store(
    vector_store: Any,
    documents: Sequence[Document],
    *,
    batch_size: int,
) -> int:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    added = 0
    total = len(documents)
    for start in range(0, total, batch_size):
        batch = list(documents[start : start + batch_size])
        vector_store.add_documents(batch, ids=[document.id for document in batch])
        added += len(batch)
        print(f"Indexed {added:,}/{total:,}")
    return added


def write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(f"{path.suffix}.part")
    partial.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    partial.replace(path)


def read_checkpoint(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_checkpoint(path: Path, checkpoint: Mapping[str, Any]) -> None:
    write_report(path, checkpoint)


def report_progress(
    *,
    started_at: float,
    batch_index: int,
    total_reviews: int,
    reviews_seen: int,
    chunks_indexed: int,
) -> None:
    elapsed = time.perf_counter() - started_at
    reviews_per_second = reviews_seen / elapsed if elapsed > 0 else 0.0
    print(
        "Indexed "
        f"{chunks_indexed:,} chunks from {reviews_seen:,}/{total_reviews:,} reviews "
        f"through review batch {batch_index:,} "
        f"({reviews_per_second:.1f} reviews/s)",
        flush=True,
    )


def run_build(
    *,
    scope: str,
    processed_root: Path,
    qdrant_url: str,
    collection: str,
    model_name: str,
    device: str,
    batch_size: int,
    review_batch_size: int,
    chunk_size: int,
    chunk_overlap: int,
    recreate: bool,
    report_output: Path,
    checkpoint_output: Path,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    print(f"Loading reviews parquet from {processed_root}", flush=True)
    parquet_path, total_reviews, review_batches = iter_review_batches(
        processed_root,
        batch_size=review_batch_size,
    )
    print(
        f"Found {total_reviews:,} reviews; review_batch_size={review_batch_size:,}",
        flush=True,
    )
    print(f"Loading tokenizer: {model_name}", flush=True)
    length_function = create_token_length_function(model_name)
    splitter = create_text_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=length_function,
    )
    print(f"Loading embedding model on {device}: {model_name}", flush=True)
    embeddings = create_embeddings(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
    )
    client = create_qdrant_client(qdrant_url)
    print(f"Preparing Qdrant collection: {collection}", flush=True)
    ensure_qdrant_collection(
        collection=collection,
        embeddings=embeddings,
        client=client,
        recreate=recreate,
    )
    print("Qdrant collection is ready", flush=True)

    checkpoint = None if recreate else read_checkpoint(checkpoint_output)
    start_batch_index = int(checkpoint.get("next_batch_index", 0)) if checkpoint else 0
    reviews_seen = int(checkpoint.get("reviews_seen", 0)) if checkpoint else 0
    indexed_documents = int(checkpoint.get("chunks_indexed", 0)) if checkpoint else 0
    split_reviews = int(checkpoint.get("split_reviews", 0)) if checkpoint else 0
    skipped_reviews = int(checkpoint.get("skipped_reviews", 0)) if checkpoint else 0
    checkpoint_parent_asins = checkpoint.get("parent_asins", []) if checkpoint else []
    rebuild_parent_asins = not isinstance(checkpoint_parent_asins, list)
    parent_asins = (
        set(checkpoint_parent_asins)
        if isinstance(checkpoint_parent_asins, list)
        else set()
    )

    review_token_lengths: list[int] = []
    chunk_token_lengths: list[int] = []
    collect_token_stats = scope != "full"

    for batch_index, record_batch in enumerate(review_batches):
        if batch_index < start_batch_index:
            if rebuild_parent_asins:
                parent_asins.update(
                    str(review["parent_asin"]) for review in record_batch.to_pylist()
                )
            continue
        reviews = record_batch.to_pylist()
        if collect_token_stats:
            review_token_lengths.extend(
                length_function(build_review_text(review)) for review in reviews
            )
        parent_asins.update(str(review["parent_asin"]) for review in reviews)
        documents, document_report = build_review_documents(
            reviews,
            split_text=splitter.split_text,
        )
        if collect_token_stats:
            chunk_token_lengths.extend(
                length_function(document.metadata["chunk_text"])
                for document in documents
            )
        def chunk_progress(batch_chunks_indexed: int) -> None:
            print(
                "Upserted "
                f"{indexed_documents + batch_chunks_indexed:,} chunks "
                f"while processing review batch {batch_index:,}",
                flush=True,
            )

        indexed_documents += upsert_documents_to_qdrant(
            client=client,
            collection=collection,
            embeddings=embeddings,
            documents=documents,
            batch_size=batch_size,
            progress_callback=chunk_progress,
        )
        reviews_seen += len(reviews)
        split_reviews += document_report["split_reviews"]
        skipped_reviews += document_report["skipped_reviews"]
        progress = BuildProgress(
            batches_processed=batch_index + 1,
            reviews_seen=reviews_seen,
            chunks_indexed=indexed_documents,
            split_reviews=split_reviews,
            skipped_reviews=skipped_reviews,
            parent_asin_count=len(parent_asins),
        )
        checkpoint_data = {
            "scope": scope,
            "processed_root": str(processed_root),
            "parquet_path": str(parquet_path),
            "collection": collection,
            "model_name": model_name,
            "batch_size": batch_size,
            "review_batch_size": review_batch_size,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "next_batch_index": batch_index + 1,
            **asdict(progress),
            "parent_asins": sorted(parent_asins),
        }
        write_checkpoint(checkpoint_output, checkpoint_data)
        write_report(
            report_output,
            {
                **checkpoint_data,
                "total_reviews": total_reviews,
                "elapsed_seconds": time.perf_counter() - started_at,
            },
        )
        report_progress(
            started_at=started_at,
            batch_index=batch_index,
            total_reviews=total_reviews,
            reviews_seen=reviews_seen,
            chunks_indexed=indexed_documents,
        )

    collection_info = client.get_collection(collection)
    elapsed_seconds = time.perf_counter() - started_at
    report = {
        "scope": scope,
        "processed_root": str(processed_root),
        "parquet_path": str(parquet_path),
        "reviews": reviews_seen,
        "total_reviews": total_reviews,
        "review_parent_asins": len(parent_asins),
        "chunks": indexed_documents,
        "indexed_documents": indexed_documents,
        "split_reviews": split_reviews,
        "skipped_reviews": skipped_reviews,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_model": model_name,
        "device": str(resolve_device(device)),
        "batch_size": batch_size,
        "review_batch_size": review_batch_size,
        "qdrant": {
            "collection": collection,
            "points_count": collection_info.points_count,
        },
        "elapsed_seconds": elapsed_seconds,
    }
    if collect_token_stats:
        report["review_token_lengths"] = asdict(
            summarize_token_lengths(review_token_lengths)
        )
        report["chunk_token_lengths"] = asdict(
            summarize_token_lengths(chunk_token_lengths)
        )
    write_report(report_output, report)
    return report


def main() -> None:
    args = parse_args()
    settings = Settings()
    processed_root = args.processed_root or PROCESSED_ROOTS[args.scope]
    collection = args.collection or collection_name(args.scope, args.model_name)
    report_output = (
        args.report_output
        or EVIDENCE_DENSE_ARTIFACT_ROOT
        / args.scope
        / "build_report.json"
    )
    checkpoint_output = (
        args.checkpoint_output
        or EVIDENCE_DENSE_ARTIFACT_ROOT
        / args.scope
        / "checkpoint.json"
    )
    report = run_build(
        scope=args.scope,
        processed_root=processed_root,
        qdrant_url=args.qdrant_url or settings.qdrant_url,
        collection=collection,
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        review_batch_size=args.review_batch_size,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        recreate=args.recreate,
        report_output=report_output,
        checkpoint_output=checkpoint_output,
    )
    print(f"Wrote report: {report_output}")


if __name__ == "__main__":
    main()
