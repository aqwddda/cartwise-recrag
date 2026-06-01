"""Interactively compare stage-six dense retrievers without reloading models."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cartwise.core.config import Settings  # noqa: E402
from cartwise.core.llm import QueryTranslationError, create_query_translator  # noqa: E402
from cartwise.retrieval.dense import (  # noqa: E402
    DENSE_MODEL_SPECS,
    DenseRetriever,
    collection_name,
    create_qdrant_client,
    load_dense_encoder,
)


EXIT_COMMANDS = {":exit", ":quit", "exit", "quit"}


class LazySettingsQueryTranslator:
    """Delay external LLM client creation until the first Chinese query."""

    def __init__(self) -> None:
        self._translator = None

    def translate(self, query: str) -> str:
        if self._translator is None:
            self._translator = create_query_translator(Settings())
        return self._translator.translate(query)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("dev", "full"), default="full")
    parser.add_argument("--qdrant-url")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=DENSE_MODEL_SPECS,
        default=list(DENSE_MODEL_SPECS),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def load_retrievers(
    *,
    scope: str,
    qdrant_url: str,
    model_keys: list[str],
    device: str,
) -> dict[str, DenseRetriever]:
    client = create_qdrant_client(qdrant_url)
    translator = LazySettingsQueryTranslator()
    retrievers: dict[str, DenseRetriever] = {}
    for model_key in model_keys:
        collection = collection_name(scope, model_key)
        info = client.get_collection(collection)
        print(
            f"Loading {model_key}: {DENSE_MODEL_SPECS[model_key].model_name} "
            f"({info.points_count:,} indexed products)"
        )
        encoder = load_dense_encoder(model_key, device=device)
        retrievers[model_key] = DenseRetriever(
            client,
            collection=collection,
            encoder=encoder,
            translator=translator,
        )
    return retrievers


def search_all(
    retrievers: Mapping[str, DenseRetriever],
    query: str,
    *,
    top_k: int,
) -> dict[str, list[dict[str, Any]]]:
    return {
        model_key: retriever.search(query, k=top_k)
        for model_key, retriever in retrievers.items()
    }


def print_results(
    query: str,
    results_by_model: Mapping[str, list[Mapping[str, Any]]],
) -> None:
    for model_key, results in results_by_model.items():
        print(f"\n[{model_key}] {query}")
        if results and results[0]["retrieval_query"] != query:
            print(f"Translated query: {results[0]['retrieval_query']}")
        for rank, result in enumerate(results, start=1):
            print(
                f"{rank}. {result['parent_asin']} "
                f"{result['dense_score']:.4f} {result.get('title')}"
            )


def interactive_loop(
    retrievers: Mapping[str, DenseRetriever],
    *,
    top_k: int,
) -> None:
    print("\nModels are ready. Enter a shopping query or :quit to exit.")
    while True:
        try:
            query = input("\nquery> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if query.casefold() in EXIT_COMMANDS:
            return
        if not query:
            continue
        try:
            print_results(query, search_all(retrievers, query, top_k=top_k))
        except QueryTranslationError as error:
            print(f"Translation error: {error}")


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise SystemExit("--top-k must be greater than zero")
    settings = Settings()
    retrievers = load_retrievers(
        scope=args.scope,
        qdrant_url=args.qdrant_url or settings.qdrant_url,
        model_keys=args.models,
        device=args.device,
    )
    interactive_loop(retrievers, top_k=args.top_k)


if __name__ == "__main__":
    main()
