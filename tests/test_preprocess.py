from __future__ import annotations

import csv
import gzip
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from scripts.pipeline.preprocess_amazon_reviews import (
    build_review_id,
    parse_price,
    run_preprocessing,
    validate_chronology,
)


def write_csv_gzip(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl_gzip(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row) + "\n")


def test_parse_price_handles_numbers_and_display_strings() -> None:
    assert parse_price(12.5) == 12.5
    assert parse_price("$1,299.99") == 1299.99
    assert parse_price(None) is None


def test_validate_chronology_rejects_future_training_data() -> None:
    with pytest.raises(RuntimeError, match="Chronology violation"):
        validate_chronology(
            {
                "train": {"user": [10, 30]},
                "valid": {"user": [20, 20]},
                "test": {},
            }
        )


def test_run_preprocessing_writes_linked_bounded_artifacts(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    output_root = tmp_path / "processed"
    split_root = raw_root / "benchmark" / "5core" / "last_out_w_his"
    fieldnames = ["user_id", "parent_asin", "rating", "timestamp", "history"]
    write_csv_gzip(
        split_root / "Musical_Instruments.train.csv.gz",
        [
            dict(zip(fieldnames, ["U1", "P1", "5.0", "10", ""])),
            dict(zip(fieldnames, ["U1", "P2", "4.0", "20", "P1"])),
        ],
    )
    write_csv_gzip(
        split_root / "Musical_Instruments.valid.csv.gz",
        [dict(zip(fieldnames, ["U1", "P1", "4.0", "30", "P1 P2"]))],
    )
    write_csv_gzip(
        split_root / "Musical_Instruments.test.csv.gz",
        [dict(zip(fieldnames, ["U1", "P2", "5.0", "40", "P1 P2 P1"]))],
    )
    write_jsonl_gzip(
        raw_root / "raw" / "meta_categories" / "meta_Musical_Instruments.jsonl.gz",
        [
            {"parent_asin": "P1", "title": "Item 1", "store": "Brand 1", "price": "$9.99"},
            {"parent_asin": "P2", "title": "Item 2", "details": {"Brand": "Brand 2"}},
        ],
    )
    write_jsonl_gzip(
        raw_root / "raw" / "review_categories" / "Musical_Instruments.jsonl.gz",
        [
            {"parent_asin": "P1", "asin": "A1", "user_id": "U1", "rating": 5.0, "timestamp": 11, "title": "good", "text": "great", "helpful_vote": 5, "verified_purchase": True},
            {"parent_asin": "P1", "asin": "A1", "user_id": "U2", "rating": 1.0, "timestamp": 12, "title": "bad", "text": "poor", "helpful_vote": 0, "verified_purchase": False},
            {"parent_asin": "P1", "asin": "A1", "user_id": "U3", "rating": 4.0, "timestamp": 13, "title": "ok", "text": "fine", "helpful_vote": 1, "verified_purchase": True},
            {"parent_asin": "P2", "asin": "A2", "user_id": "U1", "rating": 3.0, "timestamp": 21, "title": "ok", "text": "usable", "helpful_vote": 0, "verified_purchase": True},
        ],
    )

    stats = run_preprocessing(raw_root, output_root, max_reviews_per_item=2)

    items = pq.read_table(output_root / "items.parquet").to_pylist()
    reviews = pq.read_table(output_root / "reviews.parquet").to_pylist()
    assert {item["parent_asin"] for item in items} == {"P1", "P2"}
    assert Counter(review["parent_asin"] for review in reviews) == {"P1": 2, "P2": 1}
    assert all(review["review_id"].startswith("rvw_") for review in reviews)
    assert len({review["review_id"] for review in reviews}) == len(reviews)
    assert reviews[0]["review_id"] == build_review_id(reviews[0])
    assert any(review["rating"] == 1.0 for review in reviews)
    assert stats["reviews"]["low_rating_threshold"] == 3.0
    assert stats["reviews"]["low_rating_rows_retained"] == 2
    assert stats["interactions"]["chronology_valid"] is True
