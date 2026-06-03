"""Manually inspect LLM query intent parsing for stage-seven retrieval."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from scripts.paths import PROCESSED_ROOTS
from cartwise.core.config import Settings
from cartwise.core.llm import (
    QueryIntentError,
    QueryTranslationError,
    create_query_intent_parser,
    create_query_translator,
)
from cartwise.retrieval.filters import FilterConstraints, resolve_filter_constraints


def configure_console_output() -> None:
    """Keep Chinese query previews printable in limited Windows terminals."""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query",
        action="append",
        default=[],
        help="Query to parse. Can be passed multiple times.",
    )
    parser.add_argument(
        "--scope",
        choices=PROCESSED_ROOTS,
        default="full",
        help=(
            "Accepted for stage-seven command compatibility. The current offline "
            "mapping tables are fixed under data/processed."
        ),
    )
    return parser.parse_args()


def constraints_to_json(constraints: FilterConstraints) -> dict[str, object]:
    payload = asdict(constraints)
    for key, value in payload.items():
        if isinstance(value, tuple):
            payload[key] = list(value)
    return payload


def print_intent(query: str, intent) -> None:
    llm_intent = {
        "product_terms": list(intent.product_terms),
        "brands": list(intent.filters.brands),
        "excluded_brands": list(intent.filters.excluded_brands),
        "min_price": intent.filters.min_price,
        "max_price": intent.filters.max_price,
        "color_tags": list(intent.filters.color_tags),
        "material_tags": list(intent.filters.material_tags),
    }
    final_constraints = resolve_filter_constraints(
        product_terms=intent.product_terms,
        brands=intent.filters.brands,
        excluded_brands=intent.filters.excluded_brands,
        min_price=intent.filters.min_price,
        max_price=intent.filters.max_price,
        color_tags=intent.filters.color_tags,
        material_tags=intent.filters.material_tags,
    )
    print("\n" + "=" * 80)
    print(f"input_query: {query}")
    print(f"retrieval_search_query: {intent.search_query}")
    print("llm_structured_intent:")
    print(json.dumps(llm_intent, ensure_ascii=False, indent=2))
    print("final_filter_constraints:")
    print(
        json.dumps(
            constraints_to_json(final_constraints),
            ensure_ascii=False,
            indent=2,
        )
    )


def iter_queries(queries: list[str]):
    if queries:
        yield from queries
        return
    print("Enter shopping queries to parse. Empty line exits.")
    while True:
        query = input("> ").strip()
        if not query:
            return
        yield query


def main() -> None:
    configure_console_output()
    args = parse_args()
    settings = Settings()
    translator = create_query_translator(settings)
    parser = create_query_intent_parser(
        settings,
        translator=translator,
    )

    print(f"LLM model: {settings.llm_model}")
    print(f"Scope argument accepted for compatibility: {args.scope}")
    print(
        "Offline mappings: data/processed/item_to_categories.json, "
        "data/processed/brand_alias_to_canonical.json"
    )
    for query in iter_queries(args.query):
        try:
            print_intent(query, parser.parse(query))
        except (QueryIntentError, QueryTranslationError, ValueError) as error:
            print(f"\nERROR parsing query {query!r}: {error}", file=sys.stderr)


if __name__ == "__main__":
    main()
