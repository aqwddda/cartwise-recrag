"""Generate a Markdown data quality report from processed Parquet artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from scripts.paths import ARTIFACT_REPORTS_ROOT, PROCESSED_ROOT


DEFAULT_PROCESSED_ROOT = PROCESSED_ROOT
DEFAULT_OUTPUT = ARTIFACT_REPORTS_ROOT / "data_quality.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def ratio(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return f"{numerator / denominator:.2%}"


def is_missing(value: Any) -> bool:
    return value is None or value == "" or value == []


def missing_count(rows: list[dict[str, Any]], column: str) -> int:
    return sum(is_missing(row[column]) for row in rows)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def generate_report(processed_root: Path, output: Path) -> None:
    stats = load_json(processed_root / "preprocess_stats.json")
    items = pq.read_table(processed_root / "items.parquet").to_pylist()
    reviews = pq.read_table(processed_root / "reviews.parquet").to_pylist()
    interactions = {
        split: pq.read_table(
            processed_root / f"interactions_{split}.parquet"
        ).to_pylist()
        for split in ("train", "valid", "test")
    }

    all_interactions = [
        row for split_rows in interactions.values() for row in split_rows
    ]
    users = {row["user_id"] for row in all_interactions}
    reviews_per_item = Counter(row["parent_asin"] for row in reviews)
    item_count = len(items)
    review_count = len(reviews)
    review_catalog_rows = stats["reviews"]["catalog_source_rows"]

    lines = [
        "# Musical_Instruments Data Quality Report",
        "",
        "## Dataset Scope",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Catalog items | {item_count:,} |",
        f"| Users | {len(users):,} |",
        f"| Retained interactions | {len(all_interactions):,} |",
        f"| Retained review evidence candidates | {review_count:,} |",
        f"| Items with retained reviews | {len(reviews_per_item):,} |",
        f"| Maximum retained reviews for one item | {max(reviews_per_item.values(), default=0):,} |",
        "",
        "## Chronological Splits",
        "",
        "| Split | Source rows | Retained rows | Filtered outside catalog | Deduplicated rows |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in ("train", "valid", "test"):
        source = stats["interactions"]["source_rows"][split]
        retained = stats["interactions"]["retained_rows"][split]
        filtered = stats["interactions"]["filtered_outside_catalog_rows"][split]
        deduplicated = stats["interactions"]["deduplicated_rows"][split]
        lines.append(
            f"| {split} | {source:,} | {retained:,} | {filtered:,} | {deduplicated:,} |"
        )

    lines.extend(
        [
            "",
            f"Chronology validation passed across "
            f"{stats['interactions']['chronology_comparisons']:,} shared-user comparisons.",
            "",
            "## Missing Fields",
            "",
            "| Artifact | Field | Missing rows | Missing ratio |",
            "|---|---|---:|---:|",
        ]
    )
    for field in ("title", "brand", "price", "description"):
        missing = missing_count(items, field)
        lines.append(
            f"| items | {field} | {missing:,} | {ratio(missing, item_count)} |"
        )
    missing_review_text = stats["reviews"]["missing_text_rows"]
    lines.append(
        f"| source catalog reviews | text | {missing_review_text:,} | "
        f"{ratio(missing_review_text, review_catalog_rows)} |"
    )

    eligible_reviews = stats["reviews"]["eligible_text_rows"]
    filtered_reviews = stats["reviews"]["filtered_by_limit_or_duplicate_rows"]
    lines.extend(
        [
            "",
            "## Filtering Summary",
            "",
            "| Metric | Value | Ratio |",
            "|---|---:|---:|",
            f"| Metadata items absent from source metadata | {stats['items']['missing_metadata_items']:,} | {ratio(stats['items']['missing_metadata_items'], item_count)} |",
            f"| Duplicate metadata rows ignored | {stats['items']['duplicate_metadata_rows']:,} | n/a |",
            f"| Catalog review rows with empty text ignored | {missing_review_text:,} | {ratio(missing_review_text, review_catalog_rows)} |",
            f"| Eligible review rows removed by per-item cap or final deduplication | {filtered_reviews:,} | {ratio(filtered_reviews, eligible_reviews)} |",
            f"| Retained low/mid-rating review rows | {stats['reviews']['low_rating_rows_retained']:,} | {ratio(stats['reviews']['low_rating_rows_retained'], review_count)} |",
            "",
            "## Notes",
            "",
            "- `parent_asin` is the unified item key.",
            "- Official leave-last-out train, validation, and test files remain separate.",
            "- Training interactions are never augmented with validation or test interactions.",
            "- Review evidence candidates retain at most the configured number of reviews per item.",
        ]
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(f"{output.suffix}.part")
    partial.write_text("\n".join(lines) + "\n", encoding="utf-8")
    partial.replace(output)
    print(f"Wrote report: {output}")


def main() -> None:
    args = parse_args()
    generate_report(args.processed_root, args.output)


if __name__ == "__main__":
    main()
