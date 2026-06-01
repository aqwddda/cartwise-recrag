"""Report overlap between official ESCI examples and the CartWise catalog."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ESCI_EXAMPLES = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "amazon_esci"
    / "shopping_queries_dataset_examples.parquet"
)
DEFAULT_ITEMS = PROJECT_ROOT / "data" / "processed" / "items.parquet"
DEFAULT_REVIEWS = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "amazon_reviews_2023"
    / "raw"
    / "review_categories"
    / "Musical_Instruments.jsonl.gz"
)
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "reports" / "generated" / "esci_overlap.json"
DEFAULT_OUTPUT_MD = PROJECT_ROOT / "reports" / "generated" / "esci_overlap.md"
ESCI_COLUMNS = ("query_id", "query", "product_id", "product_locale", "esci_label", "split")
TARGET_LOCALE = "us"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--esci-examples", type=Path, default=DEFAULT_ESCI_EXAMPLES)
    parser.add_argument("--items", type=Path, default=DEFAULT_ITEMS)
    parser.add_argument("--reviews", type=Path, default=DEFAULT_REVIEWS)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser.parse_args()


def read_catalog_parent_asins(path: Path) -> set[str]:
    return set(
        pq.read_table(path, columns=["parent_asin"])
        .column("parent_asin")
        .to_pylist()
    )


def build_variant_mapping(
    path: Path, catalog_parent_asins: set[str]
) -> tuple[dict[str, str], set[str], int]:
    variant_to_parent: dict[str, str] = {}
    ambiguous_variants: set[str] = set()
    source_rows = 0
    with gzip.open(path, "rt", encoding="utf-8") as input_file:
        for line in input_file:
            source_rows += 1
            row = json.loads(line)
            parent_asin = row.get("parent_asin")
            asin = row.get("asin")
            if parent_asin not in catalog_parent_asins or not asin:
                continue
            previous = variant_to_parent.get(asin)
            if previous is not None and previous != parent_asin:
                ambiguous_variants.add(asin)
                continue
            variant_to_parent[asin] = parent_asin
    for asin in ambiguous_variants:
        variant_to_parent.pop(asin, None)
    return variant_to_parent, ambiguous_variants, source_rows


def normalize_label(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def normalize_split(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def normalize_locale(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def nested_counts(counter: Counter[tuple[str, str]]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for (first, second), count in sorted(counter.items()):
        result.setdefault(first, {})[second] = count
    return result


def candidate_count_summary(
    candidates_by_query: dict[tuple[str, str, int], set[str]],
) -> dict[str, Any]:
    counts = sorted(len(parent_asins) for parent_asins in candidates_by_query.values())
    if not counts:
        return {
            "queries": 0,
            "queries_with_at_least_1_candidate": 0,
            "queries_with_at_least_2_candidates": 0,
            "queries_with_at_least_5_candidates": 0,
            "queries_with_at_least_10_candidates": 0,
            "min_candidates": 0,
            "median_candidates": 0,
            "max_candidates": 0,
        }
    return {
        "queries": len(counts),
        "queries_with_at_least_1_candidate": sum(count >= 1 for count in counts),
        "queries_with_at_least_2_candidates": sum(count >= 2 for count in counts),
        "queries_with_at_least_5_candidates": sum(count >= 5 for count in counts),
        "queries_with_at_least_10_candidates": sum(count >= 10 for count in counts),
        "min_candidates": counts[0],
        "median_candidates": counts[len(counts) // 2],
        "max_candidates": counts[-1],
    }


def analyze_examples(
    path: Path,
    catalog_parent_asins: set[str],
    variant_to_parent: dict[str, str],
) -> dict[str, Any]:
    table = pq.read_table(path, columns=list(ESCI_COLUMNS))
    rows = table.to_pylist()
    locale_rows = 0
    matched_rows = 0
    matched_queries: set[tuple[str, str, int]] = set()
    matched_parent_asins: set[str] = set()
    matched_product_ids: set[str] = set()
    all_queries: set[tuple[str, str, int]] = set()
    all_product_ids: set[str] = set()
    rows_by_locale_split: Counter[tuple[str, str]] = Counter()
    queries_by_locale_split: dict[tuple[str, str], set[int]] = {}
    matched_rows_by_locale_split: Counter[tuple[str, str]] = Counter()
    matched_queries_by_locale_split: dict[tuple[str, str], set[int]] = {}
    matched_parent_asins_by_locale_split: dict[tuple[str, str], set[str]] = defaultdict(
        set
    )
    matched_labels: Counter[str] = Counter()
    matched_labels_by_locale_split: dict[tuple[str, str], Counter[str]] = defaultdict(
        Counter
    )
    candidates_by_query: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    unmatched_product_ids: set[str] = set()

    for row in rows:
        locale = normalize_locale(row["product_locale"])
        if locale != TARGET_LOCALE:
            continue
        locale_rows += 1
        split = normalize_split(row["split"])
        label = normalize_label(row["esci_label"])
        query_id = int(row["query_id"])
        product_id = str(row["product_id"])
        locale_split = (locale, split)
        query_key = (locale, split, query_id)
        all_queries.add(query_key)
        all_product_ids.add(product_id)
        rows_by_locale_split[locale_split] += 1
        queries_by_locale_split.setdefault(locale_split, set()).add(query_id)

        parent_asin = variant_to_parent.get(product_id)
        if parent_asin is None and product_id in catalog_parent_asins:
            parent_asin = product_id
        if parent_asin is None:
            unmatched_product_ids.add(product_id)
            continue

        matched_rows += 1
        matched_queries.add(query_key)
        matched_parent_asins.add(parent_asin)
        matched_product_ids.add(product_id)
        matched_rows_by_locale_split[locale_split] += 1
        matched_queries_by_locale_split.setdefault(locale_split, set()).add(query_id)
        matched_parent_asins_by_locale_split[locale_split].add(parent_asin)
        matched_labels[label] += 1
        matched_labels_by_locale_split[locale_split][label] += 1
        candidates_by_query[query_key].add(parent_asin)

    query_counts = Counter(
        {locale_split: len(query_ids) for locale_split, query_ids in queries_by_locale_split.items()}
    )
    matched_query_counts = Counter(
        {
            locale_split: len(query_ids)
            for locale_split, query_ids in matched_queries_by_locale_split.items()
        }
    )
    candidates_by_locale_split: dict[str, dict[str, dict[str, Any]]] = {}
    for locale_split in sorted(queries_by_locale_split):
        locale, split = locale_split
        candidates_by_locale_split.setdefault(locale, {})[split] = (
            candidate_count_summary(
                {
                    query_key: parent_asins
                    for query_key, parent_asins in candidates_by_query.items()
                    if query_key[:2] == locale_split
                }
            )
        )
    us_parent_asins_by_split = {
        split: matched_parent_asins_by_locale_split[("us", split)]
        for split in ("train", "test")
    }
    us_parent_asins = set().union(*us_parent_asins_by_split.values())
    return {
        "esci_examples": {
            "locale": TARGET_LOCALE,
            "rows": locale_rows,
            "queries": len(all_queries),
            "product_ids": len(all_product_ids),
            "rows_by_locale_split": nested_counts(rows_by_locale_split),
            "queries_by_locale_split": nested_counts(query_counts),
        },
        "matched_examples": {
            "rows": matched_rows,
            "queries": len(matched_queries),
            "product_ids": len(matched_product_ids),
            "parent_asins": len(matched_parent_asins),
            "labels": dict(sorted(matched_labels.items())),
            "labels_by_locale_split": {
                locale: {
                    split: dict(
                        sorted(matched_labels_by_locale_split[(locale, split)].items())
                    )
                    for split in sorted(
                        {
                            available_split
                            for available_locale, available_split in matched_labels_by_locale_split
                            if available_locale == locale
                        }
                    )
                }
                for locale in sorted(
                    {locale for locale, _ in matched_labels_by_locale_split}
                )
            },
            "rows_by_locale_split": nested_counts(matched_rows_by_locale_split),
            "queries_by_locale_split": nested_counts(matched_query_counts),
            "candidate_counts_by_locale_split": candidates_by_locale_split,
            "us_parent_asin_coverage": {
                "all": len(us_parent_asins),
                "train": len(us_parent_asins_by_split["train"]),
                "test": len(us_parent_asins_by_split["test"]),
            },
        },
        "unmatched_examples": {
            "product_ids": len(unmatched_product_ids),
        },
    }


def percentage(numerator: int, denominator: int) -> str:
    return f"{100 * numerator / denominator:.2f}%" if denominator else "n/a"


def render_markdown(stats: dict[str, Any]) -> str:
    catalog = stats["cartwise_catalog"]
    examples = stats["esci_examples"]
    matched = stats["matched_examples"]
    unmatched = stats["unmatched_examples"]
    us_test_candidates = matched["candidate_counts_by_locale_split"]["us"]["test"]
    us_parent_asins = matched["us_parent_asin_coverage"]
    lines = [
        "# Amazon ESCI US and CartWise Musical Instruments Overlap",
        "",
        "## CartWise catalog",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Catalog `parent_asin` values | {catalog['parent_asins']:,} |",
        f"| Raw Musical Instruments review rows | {catalog['raw_review_rows']:,} |",
        f"| Variant `asin -> parent_asin` mappings | {catalog['variant_asins']:,} |",
        f"| Ambiguous variant ASIN values excluded | {catalog['ambiguous_variant_asins']:,} |",
        "",
        "## ESCI examples",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Query-product rows | {examples['rows']:,} |",
        f"| Queries | {examples['queries']:,} |",
        f"| Product IDs | {examples['product_ids']:,} |",
        "",
        "## US exact ASIN overlap",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| CartWise catalog `parent_asin` values | {catalog['parent_asins']:,} |",
        f"| US train matched `parent_asin` values | {us_parent_asins['train']:,} |",
        f"| US train catalog coverage | {percentage(us_parent_asins['train'], catalog['parent_asins'])} |",
        f"| US test matched `parent_asin` values | {us_parent_asins['test']:,} |",
        f"| US test catalog coverage | {percentage(us_parent_asins['test'], catalog['parent_asins'])} |",
        f"| US train and test union `parent_asin` values | {us_parent_asins['all']:,} |",
        f"| US union catalog coverage | {percentage(us_parent_asins['all'], catalog['parent_asins'])} |",
        "",
        "## US test subset after ID join",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Matched query-product rows | {matched['rows_by_locale_split']['us']['test']:,} |",
        f"| Queries with at least one matched product | {us_test_candidates['queries']:,} |",
        f"| Queries with at least 2 unique parent candidates | {us_test_candidates['queries_with_at_least_2_candidates']:,} |",
        f"| Queries with at least 5 unique parent candidates | {us_test_candidates['queries_with_at_least_5_candidates']:,} |",
        f"| Queries with at least 10 unique parent candidates | {us_test_candidates['queries_with_at_least_10_candidates']:,} |",
        f"| Median unique parent candidates per matched query | {us_test_candidates['median_candidates']:,} |",
        f"| Maximum unique parent candidates for one query | {us_test_candidates['max_candidates']:,} |",
        "",
        "## Matched ESCI labels",
        "",
        "| Label | Rows |",
        "|---|---:|",
    ]
    lines.extend(
        f"| `{label}` | {count:,} |" for label, count in matched["labels"].items()
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
        "The ID join uses `ESCI.product_id -> Amazon Reviews 2023 review.asin -> "
        "CartWise parent_asin`, with a direct `product_id == parent_asin` fallback.",
        "CartWise uses only the US locale for ESCI overlap analysis and evaluation.",
        "Title matching is intentionally excluded because it cannot safely transfer "
            "query-product relevance labels.",
            "",
        ]
    )
    return "\n".join(lines)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(f"{path.suffix}.part")
    partial.write_text(text, encoding="utf-8")
    partial.replace(path)


def main() -> None:
    args = parse_args()
    catalog_parent_asins = read_catalog_parent_asins(args.items)
    variant_to_parent, ambiguous_variants, source_rows = build_variant_mapping(
        args.reviews, catalog_parent_asins
    )
    stats = {
        "cartwise_catalog": {
            "parent_asins": len(catalog_parent_asins),
            "raw_review_rows": source_rows,
            "variant_asins": len(variant_to_parent),
            "ambiguous_variant_asins": len(ambiguous_variants),
        },
        **analyze_examples(args.esci_examples, catalog_parent_asins, variant_to_parent),
    }
    write_text(args.output_json, json.dumps(stats, ensure_ascii=False, indent=2) + "\n")
    write_text(args.output_md, render_markdown(stats))
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"[done] {args.output_json}")
    print(f"[done] {args.output_md}")


if __name__ == "__main__":
    main()
