from __future__ import annotations

import csv
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.evaluate import PROJECT_ROOT, evaluate_popularity, resolve_paths


def write_interactions(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_evaluate_popularity_writes_validation_and_test_metrics(tmp_path: Path) -> None:
    processed_root = tmp_path / "processed"
    output = tmp_path / "reports" / "popularity_metrics.csv"
    write_interactions(
        processed_root / "interactions_train.parquet",
        [
            {"user_id": "U1", "parent_asin": "P1"},
            {"user_id": "U2", "parent_asin": "P1"},
            {"user_id": "U2", "parent_asin": "P2"},
            {"user_id": "U3", "parent_asin": "P3"},
        ],
    )
    write_interactions(
        processed_root / "interactions_valid.parquet",
        [{"user_id": "U1", "parent_asin": "P2"}],
    )
    write_interactions(
        processed_root / "interactions_test.parquet",
        [{"user_id": "U1", "parent_asin": "P3"}],
    )

    evaluate_popularity(processed_root, output, k=10)

    with output.open(newline="", encoding="utf-8") as input_file:
        rows = list(csv.DictReader(input_file))
    assert rows == [
        {
            "model": "popularity",
            "split": "valid",
            "users": "1",
            "Recall@10": "1.000000",
            "NDCG@10": "1.000000",
            "HitRate@10": "1.000000",
        },
        {
            "model": "popularity",
            "split": "test",
            "users": "1",
            "Recall@10": "1.000000",
            "NDCG@10": "1.000000",
            "HitRate@10": "1.000000",
        },
    ]


def test_resolve_paths_uses_scope_defaults() -> None:
    assert resolve_paths("dev") == (
        PROJECT_ROOT / "data" / "processed" / "dev",
        PROJECT_ROOT / "reports" / "metrics" / "dev" / "popularity.csv",
    )
    assert resolve_paths("full") == (
        PROJECT_ROOT / "data" / "processed",
        PROJECT_ROOT / "reports" / "metrics" / "full" / "popularity.csv",
    )


def test_resolve_paths_allows_explicit_overrides(tmp_path: Path) -> None:
    processed_root = tmp_path / "custom-processed"
    output = tmp_path / "custom-reports" / "popularity.csv"

    assert resolve_paths("full", processed_root, output) == (processed_root, output)
