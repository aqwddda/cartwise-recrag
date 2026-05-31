"""Preprocess the Amazon Reviews 2023 Musical_Instruments dataset."""

from __future__ import annotations

import argparse
import csv
import gzip
import heapq
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "amazon_reviews_2023"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed"
SPLIT_PATHS = {
    "train": "benchmark/5core/last_out_w_his/Musical_Instruments.train.csv.gz",
    "valid": "benchmark/5core/last_out_w_his/Musical_Instruments.valid.csv.gz",
    "test": "benchmark/5core/last_out_w_his/Musical_Instruments.test.csv.gz",
}
METADATA_PATH = "raw/meta_categories/meta_Musical_Instruments.jsonl.gz"
REVIEWS_PATH = "raw/review_categories/Musical_Instruments.jsonl.gz"
PRICE_PATTERN = re.compile(r"\d[\d,]*(?:\.\d+)?")

ITEM_SCHEMA = pa.schema(
    [
        ("parent_asin", pa.string()),
        ("title", pa.string()),
        ("brand", pa.string()),
        ("price", pa.float64()),
        ("main_category", pa.string()),
        ("categories", pa.list_(pa.string())),
        ("description", pa.string()),
        ("features", pa.list_(pa.string())),
        ("details_json", pa.string()),
        ("bought_together", pa.list_(pa.string())),
    ]
)
INTERACTION_SCHEMA = pa.schema(
    [
        ("user_id", pa.string()),
        ("parent_asin", pa.string()),
        ("rating", pa.float64()),
        ("timestamp", pa.int64()),
    ]
)
REVIEW_SCHEMA = pa.schema(
    [
        ("parent_asin", pa.string()),
        ("asin", pa.string()),
        ("user_id", pa.string()),
        ("rating", pa.float64()),
        ("timestamp", pa.int64()),
        ("title", pa.string()),
        ("text", pa.string()),
        ("helpful_vote", pa.int64()),
        ("verified_purchase", pa.bool_()),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--max-items",
        type=int,
        help="Keep only the N most frequent training items for a development run.",
    )
    parser.add_argument("--max-reviews-per-item", type=int, default=10)
    return parser.parse_args()


def read_csv_rows(path: Path) -> Iterator[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as input_file:
        yield from csv.DictReader(input_file)


def read_jsonl_rows(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as input_file:
        for line in input_file:
            yield json.loads(line)


def normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def normalize_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [text for item in value if (text := normalize_text(item)) is not None]


def parse_price(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = PRICE_PATTERN.search(str(value))
    if match is None:
        return None
    return float(match.group(0).replace(",", ""))


def write_parquet_rows(
    path: Path,
    rows: Iterable[dict[str, Any]],
    schema: pa.Schema,
    batch_size: int = 10_000,
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(f"{path.suffix}.part")
    if partial.exists():
        partial.unlink()

    count = 0
    writer: pq.ParquetWriter | None = None
    batch: list[dict[str, Any]] = []
    try:
        for row in rows:
            batch.append(row)
            if len(batch) >= batch_size:
                writer = writer or pq.ParquetWriter(partial, schema)
                writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                count += len(batch)
                batch.clear()
        if batch or writer is None:
            writer = writer or pq.ParquetWriter(partial, schema)
            if batch:
                writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                count += len(batch)
    finally:
        if writer is not None:
            writer.close()

    partial.replace(path)
    return count


def scan_interaction_catalog(
    raw_root: Path,
) -> tuple[set[str], Counter[str], dict[str, int]]:
    items: set[str] = set()
    training_counts: Counter[str] = Counter()
    source_rows: dict[str, int] = {}
    for split, relative_path in SPLIT_PATHS.items():
        count = 0
        for row in read_csv_rows(raw_root / relative_path):
            parent_asin = row["parent_asin"]
            items.add(parent_asin)
            if split == "train":
                training_counts[parent_asin] += 1
            count += 1
        source_rows[split] = count
    return items, training_counts, source_rows


def select_catalog_items(
    items: set[str], training_counts: Counter[str], max_items: int | None
) -> set[str]:
    if max_items is None:
        return items
    if max_items <= 0:
        raise ValueError("--max-items must be greater than zero")
    ranked_items = sorted(items, key=lambda item: (-training_counts[item], item))
    return set(ranked_items[:max_items])


def update_timestamp_bounds(
    bounds: dict[str, list[int]], user_id: str, timestamp: int
) -> None:
    if user_id not in bounds:
        bounds[user_id] = [timestamp, timestamp]
        return
    bounds[user_id][0] = min(bounds[user_id][0], timestamp)
    bounds[user_id][1] = max(bounds[user_id][1], timestamp)


def validate_chronology(bounds_by_split: Mapping[str, Mapping[str, list[int]]]) -> int:
    comparisons = 0
    pairs = (("train", "valid"), ("valid", "test"), ("train", "test"))
    for earlier, later in pairs:
        shared_users = bounds_by_split[earlier].keys() & bounds_by_split[later].keys()
        for user_id in shared_users:
            earlier_max = bounds_by_split[earlier][user_id][1]
            later_min = bounds_by_split[later][user_id][0]
            if earlier_max > later_min:
                raise RuntimeError(
                    f"Chronology violation for {user_id}: "
                    f"{earlier} timestamp {earlier_max} exceeds "
                    f"{later} timestamp {later_min}"
                )
            comparisons += 1
    return comparisons


def write_interactions(
    raw_root: Path, output_root: Path, selected_items: set[str]
) -> tuple[dict[str, Any], set[str]]:
    retained_rows: dict[str, int] = {}
    deduplicated_rows: dict[str, int] = {}
    filtered_rows: dict[str, int] = {}
    bounds_by_split: dict[str, dict[str, list[int]]] = {
        split: {} for split in SPLIT_PATHS
    }
    users: set[str] = set()
    seen: set[tuple[str, str, float, int]] = set()

    for split, relative_path in SPLIT_PATHS.items():
        split_deduplicated = 0
        split_filtered = 0

        def rows() -> Iterator[dict[str, Any]]:
            nonlocal split_deduplicated, split_filtered
            for source_row in read_csv_rows(raw_root / relative_path):
                parent_asin = source_row["parent_asin"]
                if parent_asin not in selected_items:
                    split_filtered += 1
                    continue
                row = {
                    "user_id": source_row["user_id"],
                    "parent_asin": parent_asin,
                    "rating": float(source_row["rating"]),
                    "timestamp": int(source_row["timestamp"]),
                }
                signature = (
                    row["user_id"],
                    row["parent_asin"],
                    row["rating"],
                    row["timestamp"],
                )
                if signature in seen:
                    split_deduplicated += 1
                    continue
                seen.add(signature)
                users.add(row["user_id"])
                update_timestamp_bounds(
                    bounds_by_split[split], row["user_id"], row["timestamp"]
                )
                yield row

        retained_rows[split] = write_parquet_rows(
            output_root / f"interactions_{split}.parquet", rows(), INTERACTION_SCHEMA
        )
        deduplicated_rows[split] = split_deduplicated
        filtered_rows[split] = split_filtered

    chronology_comparisons = validate_chronology(bounds_by_split)
    return (
        {
            "retained_rows": retained_rows,
            "deduplicated_rows": deduplicated_rows,
            "filtered_outside_catalog_rows": filtered_rows,
            "users": len(users),
            "chronology_comparisons": chronology_comparisons,
            "chronology_valid": True,
        },
        users,
    )


def build_item_row(source_row: Mapping[str, Any]) -> dict[str, Any]:
    details = source_row.get("details")
    details = details if isinstance(details, dict) else {}
    description = "\n".join(normalize_string_list(source_row.get("description")))
    brand = normalize_text(details.get("Brand")) or normalize_text(
        source_row.get("store")
    )
    return {
        "parent_asin": source_row["parent_asin"],
        "title": normalize_text(source_row.get("title")),
        "brand": brand,
        "price": parse_price(source_row.get("price")),
        "main_category": normalize_text(source_row.get("main_category")),
        "categories": normalize_string_list(source_row.get("categories")),
        "description": description or None,
        "features": normalize_string_list(source_row.get("features")),
        "details_json": json.dumps(details, ensure_ascii=False, sort_keys=True),
        "bought_together": normalize_string_list(source_row.get("bought_together")),
    }


def write_items(
    raw_root: Path, output_root: Path, selected_items: set[str]
) -> dict[str, int]:
    source_rows = 0
    duplicate_metadata_rows = 0
    item_rows: dict[str, dict[str, Any]] = {}
    for source_row in read_jsonl_rows(raw_root / METADATA_PATH):
        source_rows += 1
        parent_asin = source_row["parent_asin"]
        if parent_asin not in selected_items:
            continue
        if parent_asin in item_rows:
            duplicate_metadata_rows += 1
            continue
        item_rows[parent_asin] = build_item_row(source_row)

    missing_metadata_items = selected_items - item_rows.keys()
    for parent_asin in missing_metadata_items:
        item_rows[parent_asin] = build_item_row({"parent_asin": parent_asin})

    write_parquet_rows(
        output_root / "items.parquet",
        (item_rows[parent_asin] for parent_asin in sorted(item_rows)),
        ITEM_SCHEMA,
    )
    return {
        "catalog": len(selected_items),
        "source_metadata_rows": source_rows,
        "matched_metadata_items": len(selected_items) - len(missing_metadata_items),
        "missing_metadata_items": len(missing_metadata_items),
        "duplicate_metadata_rows": duplicate_metadata_rows,
    }


def push_ranked_review(
    heap: list[tuple[tuple[int, int, int], int, dict[str, Any]]],
    row: dict[str, Any],
    sequence: int,
    capacity: int,
) -> None:
    priority = (
        int(row["verified_purchase"]),
        row["helpful_vote"],
        row["timestamp"],
    )
    entry = (priority, sequence, row)
    if len(heap) < capacity:
        heapq.heappush(heap, entry)
    elif entry[:2] > heap[0][:2]:
        heapq.heapreplace(heap, entry)


def review_signature(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row["parent_asin"],
        row["asin"],
        row["user_id"],
        row["timestamp"],
        row["title"],
        row["text"],
    )


def select_reviews(
    raw_root: Path, selected_items: set[str], max_reviews_per_item: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if max_reviews_per_item <= 0:
        raise ValueError("--max-reviews-per-item must be greater than zero")

    general_heaps: dict[str, list[tuple[tuple[int, int, int], int, dict[str, Any]]]] = (
        defaultdict(list)
    )
    low_rating_heaps: dict[
        str, list[tuple[tuple[int, int, int], int, dict[str, Any]]]
    ] = defaultdict(list)
    low_rating_capacity = min(2, max_reviews_per_item)
    source_rows = 0
    catalog_source_rows = 0
    missing_text_rows = 0
    eligible_text_rows = 0

    for sequence, source_row in enumerate(read_jsonl_rows(raw_root / REVIEWS_PATH)):
        source_rows += 1
        parent_asin = source_row["parent_asin"]
        if parent_asin not in selected_items:
            continue
        catalog_source_rows += 1
        text = normalize_text(source_row.get("text"))
        if text is None:
            missing_text_rows += 1
            continue
        eligible_text_rows += 1
        row = {
            "parent_asin": parent_asin,
            "asin": normalize_text(source_row.get("asin")),
            "user_id": normalize_text(source_row.get("user_id")),
            "rating": float(source_row["rating"]),
            "timestamp": int(source_row["timestamp"]),
            "title": normalize_text(source_row.get("title")),
            "text": text,
            "helpful_vote": int(source_row.get("helpful_vote") or 0),
            "verified_purchase": bool(source_row.get("verified_purchase")),
        }
        push_ranked_review(
            general_heaps[parent_asin], row, sequence, max_reviews_per_item
        )
        if row["rating"] <= 2.0:
            push_ranked_review(
                low_rating_heaps[parent_asin], row, sequence, low_rating_capacity
            )

    selected_reviews: list[dict[str, Any]] = []
    for parent_asin in sorted(selected_items):
        candidates = sorted(low_rating_heaps[parent_asin], reverse=True)
        candidates.extend(sorted(general_heaps[parent_asin], reverse=True))
        signatures: set[tuple[Any, ...]] = set()
        for _, _, row in candidates:
            signature = review_signature(row)
            if signature in signatures:
                continue
            signatures.add(signature)
            selected_reviews.append(row)
            if len(signatures) == max_reviews_per_item:
                break

    return selected_reviews, {
        "source_rows": source_rows,
        "catalog_source_rows": catalog_source_rows,
        "missing_text_rows": missing_text_rows,
        "eligible_text_rows": eligible_text_rows,
        "retained_rows": len(selected_reviews),
        "filtered_by_limit_or_duplicate_rows": eligible_text_rows
        - len(selected_reviews),
        "items_with_reviews": len({row["parent_asin"] for row in selected_reviews}),
        "low_rating_rows_retained": sum(
            row["rating"] <= 2.0 for row in selected_reviews
        ),
    }


def write_stats(output_root: Path, stats: Mapping[str, Any]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "preprocess_stats.json"
    partial = path.with_suffix(f"{path.suffix}.part")
    partial.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    partial.replace(path)


def run_preprocessing(
    raw_root: Path = DEFAULT_RAW_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    max_items: int | None = None,
    max_reviews_per_item: int = 10,
) -> dict[str, Any]:
    print("[1/5] Scanning official interaction splits")
    all_items, training_counts, source_interaction_rows = scan_interaction_catalog(
        raw_root
    )
    selected_items = select_catalog_items(all_items, training_counts, max_items)
    print(f"      Selected {len(selected_items):,} of {len(all_items):,} catalog items")

    print("[2/5] Writing chronological interaction Parquet files")
    interaction_stats, _ = write_interactions(raw_root, output_root, selected_items)
    interaction_stats["source_rows"] = source_interaction_rows

    print("[3/5] Joining product metadata")
    item_stats = write_items(raw_root, output_root, selected_items)

    print("[4/5] Selecting review evidence candidates")
    review_rows, review_stats = select_reviews(
        raw_root, selected_items, max_reviews_per_item
    )
    write_parquet_rows(output_root / "reviews.parquet", review_rows, REVIEW_SCHEMA)

    stats = {
        "dataset": "Amazon Reviews 2023 Musical_Instruments",
        "max_items": max_items,
        "max_reviews_per_item": max_reviews_per_item,
        "items": item_stats,
        "interactions": interaction_stats,
        "reviews": review_stats,
    }
    print("[5/5] Writing preprocessing statistics")
    write_stats(output_root, stats)
    print(f"      Output: {output_root}")
    return stats


def main() -> None:
    args = parse_args()
    run_preprocessing(
        raw_root=args.raw_root,
        output_root=args.output_root,
        max_items=args.max_items,
        max_reviews_per_item=args.max_reviews_per_item,
    )


if __name__ == "__main__":
    main()
