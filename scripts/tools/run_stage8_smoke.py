"""Run a read-only stage-eight retrieval and explanation smoke test."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from openai import OpenAI

from cartwise.core.config import Settings
from cartwise.evidence.rag import (
    EvidenceRagConfig,
    OpenAICompatibleExplanationGenerator,
    QdrantReviewEvidenceRetriever,
    explain_candidates,
)
from cartwise.retrieval.bm25 import BM25Index, BM25Retriever
from cartwise.retrieval.dense import DenseRetriever
from cartwise.retrieval.dense import collection_name as dense_collection_name
from cartwise.retrieval.dense import create_qdrant_client, load_dense_encoder
from cartwise.retrieval.filters import FilterConstraints
from cartwise.retrieval.fusion import BM25_CHANNEL, DENSE_CHANNEL, FusionConfig, fuse_candidates
from scripts.paths import ARTIFACT_REPORTS_ROOT, PRODUCT_BM25_ARTIFACT_ROOT, PROCESSED_ROOTS
from scripts.pipeline.build_evidence_index import collection_name as evidence_collection_name
from scripts.pipeline.build_evidence_index import DEFAULT_REVIEW_EMBEDDING_MODEL
from scripts.tools.item_metadata import load_items_by_parent_asin


DEFAULT_QUERY = "guitar tuner for beginners"
DEFAULT_OUTPUT_ROOT = ARTIFACT_REPORTS_ROOT / "stage8_smoke"


class RecordingGenerator:
    def __init__(self, wrapped: OpenAICompatibleExplanationGenerator) -> None:
        self.wrapped = wrapped
        self.last_prompt: str | None = None
        self.last_content: str | None = None
        self.records: list[dict[str, str]] = []

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        self.last_content = self.wrapped.generate(prompt)
        self.records.append(
            {
                "prompt": prompt,
                "content": self.last_content,
            }
        )
        return self.last_content


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=PROCESSED_ROOTS, default="full")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dense-k", type=int, default=10)
    parser.add_argument("--bm25-k", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--qdrant-url")
    parser.add_argument("--show-llm-raw", action="store_true")
    parser.add_argument("--include-prompts", action="store_true")
    parser.add_argument("--snippet-chars", type=int, default=600)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def safe_stem(value: str, *, max_length: int = 72) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    return (stem or "query")[:max_length]


def json_default(value: Any) -> str:
    return str(value)


def evidence_to_dict(evidence: Any) -> dict[str, Any]:
    return asdict(evidence)


def ordered_evidence(explanation: Any) -> tuple[list[Any], list[Any]]:
    cited: list[Any] = []
    additional: list[Any] = []
    seen_indexes: set[int] = set()

    for review_id in explanation.cited_review_ids:
        for index, evidence in enumerate(explanation.evidence):
            if index in seen_indexes:
                continue
            if evidence.review_id == review_id:
                cited.append(evidence)
                seen_indexes.add(index)

    for index, evidence in enumerate(explanation.evidence):
        if index not in seen_indexes:
            additional.append(evidence)
    return cited, additional


def compact_text(value: str | None, *, limit: int) -> str:
    if not value:
        return ""
    text = " ".join(value.split())
    if limit <= 0 or len(text) <= limit:
        return text
    return f"{text[:limit]}..."


def format_evidence_block(title: str, evidence_items: list[Any], *, full: bool, limit: int) -> str:
    lines = [f"{title}:"]
    if not evidence_items:
        lines.append("  none")
        return "\n".join(lines)

    for evidence in evidence_items:
        chunk_text = evidence.chunk_text if full else compact_text(evidence.chunk_text, limit=limit)
        review_text = evidence.text if full else compact_text(evidence.text, limit=limit)
        lines.extend(
            [
                f"  - review_id: {evidence.review_id}",
                f"    chunk_id: {evidence.chunk_id}",
                f"    rating: {evidence.rating}",
                f"    score: {evidence.score}",
                f"    title: {evidence.title}",
                f"    helpful_vote: {evidence.helpful_vote}",
                f"    verified_purchase: {evidence.verified_purchase}",
                f"    timestamp: {evidence.timestamp}",
                "    chunk_text:",
                indent_block(chunk_text, spaces=6),
            ]
        )
        if full and review_text:
            lines.extend(
                [
                    "    full_review_text:",
                    indent_block(review_text, spaces=6),
                ]
            )
    return "\n".join(lines)


def indent_block(value: str | None, *, spaces: int) -> str:
    prefix = " " * spaces
    text = value or ""
    return "\n".join(f"{prefix}{line}" for line in text.splitlines()) or prefix


def build_report(
    *,
    args: argparse.Namespace,
    qdrant_url: str,
    dense_candidates: list[dict[str, Any]],
    bm25_candidates: list[dict[str, Any]],
    fusion_results: list[dict[str, Any]],
    explanations: list[Any],
    llm_records: list[dict[str, str]],
) -> dict[str, Any]:
    by_parent_asin = {explanation.parent_asin: explanation for explanation in explanations}
    products: list[dict[str, Any]] = []
    for index, result in enumerate(fusion_results):
        explanation = by_parent_asin[result["parent_asin"]]
        cited, additional = ordered_evidence(explanation)
        llm_record = llm_records[index] if index < len(llm_records) else None
        products.append(
            {
                "rank": result["rank"],
                "parent_asin": result["parent_asin"],
                "item": result["item"],
                "sources": result.get("sources", []),
                "fusion_score": result["fusion_score"],
                "explanation": {
                    "fallback": explanation.fallback,
                    "reason": explanation.reason,
                    "potential_cons": explanation.potential_cons,
                    "cited_review_ids": list(explanation.cited_review_ids),
                },
                "cited_evidence": [evidence_to_dict(evidence) for evidence in cited],
                "additional_evidence": [evidence_to_dict(evidence) for evidence in additional],
                "llm_output": llm_record["content"] if llm_record else None,
                "llm_prompt": llm_record["prompt"] if args.include_prompts and llm_record else None,
            }
        )

    return {
        "metadata": {
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "query": args.query,
            "scope": args.scope,
            "top_k": args.top_k,
            "dense_k": args.dense_k,
            "bm25_k": args.bm25_k,
            "device": args.device,
            "qdrant_url": qdrant_url,
            "dense_candidate_count": len(dense_candidates),
            "bm25_candidate_count": len(bm25_candidates),
            "include_prompts": args.include_prompts,
        },
        "products": products,
    }


def format_report_text(
    report: dict[str, Any],
    *,
    full: bool,
    snippet_chars: int,
    include_llm_details: bool = True,
) -> str:
    metadata = report["metadata"]
    lines = [
        "Stage 8 smoke report",
        f"Generated at: {metadata['generated_at']}",
        f"Query: {metadata['query']}",
        f"Scope: {metadata['scope']}",
        (
            "Dense candidates: "
            f"{metadata['dense_candidate_count']} | BM25 candidates: {metadata['bm25_candidate_count']}"
        ),
        "",
        "=== Final recalled products with LLM explanations ===",
    ]
    for product in report["products"]:
        item = product["item"]
        explanation = product["explanation"]
        lines.extend(
            [
                "",
                f"#{product['rank']} {product['parent_asin']}",
                f"Title: {item.get('title')}",
                f"Brand: {item.get('brand')}",
                f"Price: {item.get('price')}",
                f"Sources: {', '.join(product.get('sources', []))}",
                f"Fusion score: {product['fusion_score']:.6f}",
                f"Fallback explanation: {explanation['fallback']}",
                f"Reason: {explanation['reason']}",
                f"Potential cons: {explanation['potential_cons']}",
                f"Cited review IDs: {', '.join(explanation['cited_review_ids']) or 'none'}",
                format_evidence_block(
                    "Cited evidence",
                    [SimpleEvidence(evidence) for evidence in product["cited_evidence"]],
                    full=full,
                    limit=snippet_chars,
                ),
                format_evidence_block(
                    "Additional retrieved evidence",
                    [SimpleEvidence(evidence) for evidence in product["additional_evidence"]],
                    full=full,
                    limit=snippet_chars,
                ),
            ]
        )
        if include_llm_details and product.get("llm_output"):
            lines.extend(
                [
                    "Raw LLM output:",
                    indent_block(product["llm_output"], spaces=2),
                ]
            )
        if include_llm_details and product.get("llm_prompt"):
            lines.extend(
                [
                    "LLM prompt:",
                    indent_block(product["llm_prompt"], spaces=2),
                ]
            )
    return "\n".join(lines) + "\n"


class SimpleEvidence:
    def __init__(self, values: dict[str, Any]) -> None:
        self.parent_asin = values.get("parent_asin")
        self.review_id = values.get("review_id")
        self.chunk_id = values.get("chunk_id")
        self.rating = values.get("rating")
        self.title = values.get("title")
        self.text = values.get("text")
        self.chunk_text = values.get("chunk_text")
        self.helpful_vote = values.get("helpful_vote")
        self.verified_purchase = values.get("verified_purchase")
        self.timestamp = values.get("timestamp")
        self.score = values.get("score")


def write_report_files(report: dict[str, Any], output_dir: Path, *, query: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{timestamp}_{safe_stem(query)}"
    json_path = output_dir / f"{stem}.json"
    text_path = output_dir / f"{stem}.txt"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    text_path.write_text(
        format_report_text(report, full=True, snippet_chars=0),
        encoding="utf-8",
    )
    return json_path, text_path


def main() -> None:
    args = parse_args()
    settings = Settings()
    if settings.llm_api_key is None:
        raise SystemExit("DeepSeek/OpenAI-compatible API key is not configured")

    qdrant_url = args.qdrant_url or settings.qdrant_url
    processed_root = PROCESSED_ROOTS[args.scope]
    items_by_parent_asin = load_items_by_parent_asin(processed_root / "items.parquet")

    client = create_qdrant_client(qdrant_url)
    encoder = load_dense_encoder("e5", device=args.device)
    dense_retriever = DenseRetriever(
        client,
        collection=dense_collection_name(args.scope, "e5"),
        encoder=encoder,
    )
    bm25_retriever = BM25Retriever(
        BM25Index.load(PRODUCT_BM25_ARTIFACT_ROOT / args.scope / "bm25.json.gz"),
    )

    dense_candidates = [
        {
            "channel": DENSE_CHANNEL,
            "rank": rank,
            "parent_asin": result["parent_asin"],
            "score": result["dense_score"],
            "score_type": "dense_score",
            "item": dict(items_by_parent_asin[result["parent_asin"]]),
            "retrieval_query": result["retrieval_query"],
            "document": result.get("document"),
        }
        for rank, result in enumerate(
            dense_retriever.search(args.query, k=args.dense_k),
            start=1,
        )
        if result["parent_asin"] in items_by_parent_asin
    ]
    bm25_candidates = [
        {
            "channel": BM25_CHANNEL,
            "rank": rank,
            "parent_asin": result["parent_asin"],
            "score": result["bm25_score"],
            "score_type": "bm25_score",
            "item": dict(items_by_parent_asin[result["parent_asin"]]),
            "retrieval_query": result["retrieval_query"],
            "document": result.get("document"),
        }
        for rank, result in enumerate(
            bm25_retriever.search(args.query, k=args.bm25_k),
            start=1,
        )
        if result["parent_asin"] in items_by_parent_asin
    ]
    fusion = fuse_candidates(
        {
            DENSE_CHANNEL: dense_candidates,
            BM25_CHANNEL: bm25_candidates,
        },
        FilterConstraints(),
        config=FusionConfig(
            dense_k=args.dense_k,
            bm25_k=args.bm25_k,
            final_top_k=args.top_k,
        ),
        known_user=False,
    )

    evidence_retriever = QdrantReviewEvidenceRetriever(
        client,
        collection=evidence_collection_name(args.scope, DEFAULT_REVIEW_EMBEDDING_MODEL),
        encoder=encoder,
    )
    http_client = httpx.Client(
        proxy=settings.external_https_proxy,
        trust_env=False,
    )
    llm_client = OpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        timeout=settings.llm_timeout_seconds,
        http_client=http_client,
    )
    generator = RecordingGenerator(
        OpenAICompatibleExplanationGenerator(
            llm_client,
            model=settings.llm_model,
        )
    )
    evidence_config = EvidenceRagConfig(
        initial_chunk_k=10,
        final_review_k=5,
        max_candidate_chunk_k=20,
    )
    explanations = []
    for result in fusion.final_results:
        explanations.extend(
            explain_candidates(
                english_query=args.query,
                candidates=[result],
                retriever=evidence_retriever,
                generator=generator,
                config=evidence_config,
            )
        )

    report = build_report(
        args=args,
        qdrant_url=qdrant_url,
        dense_candidates=dense_candidates,
        bm25_candidates=bm25_candidates,
        fusion_results=list(fusion.final_results),
        explanations=explanations,
        llm_records=generator.records,
    )
    json_path, text_path = write_report_files(report, args.output_dir, query=args.query)

    print(
        format_report_text(
            report,
            full=False,
            snippet_chars=args.snippet_chars,
            include_llm_details=False,
        )
    )
    print(f"Full JSON report: {json_path}")
    print(f"Full text report: {text_path}")
    if args.show_llm_raw:
        print("\n=== Raw LLM outputs ===")
        for index, record in enumerate(generator.records, start=1):
            print(f"\n--- LLM call #{index} ---")
            print(record["content"])


if __name__ == "__main__":
    main()
