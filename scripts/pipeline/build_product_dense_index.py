"""Build and optionally inspect the stage-six E5 and BLaIR Qdrant indexes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


from cartwise.core.config import Settings
from cartwise.core.llm import (
    QueryTranslationError,
    contains_chinese_characters,
    create_query_translator,
)
from cartwise.retrieval.dense import (
    DENSE_MODEL_SPECS,
    DenseRetriever,
    build_qdrant_collection,
    collection_name,
    create_qdrant_client,
    load_dense_encoder,
)
from scripts.paths import PROCESSED_ROOTS, PRODUCT_DENSE_ARTIFACT_ROOT


SCOPE_PATHS = PROCESSED_ROOTS
DEFAULT_REPORT_ROOT = PRODUCT_DENSE_ARTIFACT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=SCOPE_PATHS, default="dev")
    parser.add_argument("--processed-root", type=Path)
    parser.add_argument("--qdrant-url")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=DENSE_MODEL_SPECS,
        default=list(DENSE_MODEL_SPECS),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--query-only", action="store_true")
    parser.add_argument("--query", action="append", default=[])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--report-output", type=Path)
    return parser.parse_args()


def load_items(processed_root: Path) -> list[dict[str, Any]]:
    return pq.read_table(processed_root / "items.parquet").to_pylist()


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(f"{path.suffix}.part")
    partial.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    partial.replace(path)


def run_build(
    *,
    scope: str,
    processed_root: Path,
    qdrant_url: str,
    model_keys: list[str],
    device: str,
    batch_size: int,
    recreate: bool,
    query_only: bool,
    queries: list[str],
    top_k: int,
) -> dict[str, Any]:
    if top_k <= 0:
        raise ValueError("--top-k must be greater than zero")
    if query_only and not queries:
        raise ValueError("--query-only requires at least one --query")
    items = [] if query_only else load_items(processed_root)
    client = create_qdrant_client(qdrant_url)
    report: dict[str, Any] = {
        "scope": scope,
        "items": len(items) if not query_only else None,
        "query_only": query_only,
        "collections": {},
        "queries": {},
    }
    translator = None
    if any(contains_chinese_characters(query) for query in queries):
        translator = create_query_translator(Settings())

    for model_key in model_keys:
        collection = collection_name(scope, model_key)
        print(f"Loading {model_key}: {DENSE_MODEL_SPECS[model_key].model_name}")
        encoder = load_dense_encoder(model_key, device=device)
        if query_only:
            collection_report = {
                "collection": collection,
                "model_key": encoder.key,
                "model_name": encoder.model_name,
                "points_count": client.get_collection(collection).points_count,
                "vector_size": encoder.vector_size,
            }
        else:
            print(f"Building collection: {collection}")
            collection_report = build_qdrant_collection(
                client,
                collection=collection,
                items=items,
                encoder=encoder,
                batch_size=batch_size,
                recreate=recreate,
            )
        report["collections"][model_key] = collection_report
        retriever = DenseRetriever(
            client,
            collection=collection,
            encoder=encoder,
            translator=translator,
        )
        for query in queries:
            results = retriever.search(query, k=top_k)
            report["queries"].setdefault(query, {})[model_key] = [
                {
                    "parent_asin": result["parent_asin"],
                    "score": result["dense_score"],
                    "title": result.get("title"),
                    "translated_query": result["retrieval_query"],
                }
                for result in results
            ]
            print(f"\n[{model_key}] {query}")
            for rank, result in enumerate(results, start=1):
                print(
                    f"{rank}. {result['parent_asin']} "
                    f"{result['dense_score']:.4f} {result.get('title')}"
                )
    return report


def main() -> None:
    args = parse_args()
    settings = Settings()
    processed_root = args.processed_root or SCOPE_PATHS[args.scope]
    report_output = (
        args.report_output
        or DEFAULT_REPORT_ROOT
        / args.scope
        / ("query_report.json" if args.query_only else "build_report.json")
    )
    try:
        report = run_build(
            scope=args.scope,
            processed_root=processed_root,
            qdrant_url=args.qdrant_url or settings.qdrant_url,
            model_keys=args.models,
            device=args.device,
            batch_size=args.batch_size,
            recreate=args.recreate,
            query_only=args.query_only,
            queries=args.query,
            top_k=args.top_k,
        )
    except QueryTranslationError as error:
        raise SystemExit(str(error)) from error
    write_report(report_output, report)
    print(f"Wrote report: {report_output}")


if __name__ == "__main__":
    main()
