from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.pipeline.evaluate_retrieval_audit import (
    CHANNELS,
    evaluate_retrieval_audit,
    group_scores,
    load_scores,
)


FIELDS = [
    "report_id",
    "generated_at",
    "scope",
    "query_id",
    "input",
    "retrieval_query",
    "channel",
    "rank",
    "parent_asin",
    "title",
    "brand",
    "price",
    "retrieval_score",
    "score_type",
    "human_relevance",
    "notes",
]


def write_scores(
    root: Path,
    *,
    channels: tuple[str, ...] = CHANNELS,
    query_ids: tuple[str, ...] = ("EN-01", "ZH-01"),
    k: int = 10,
    relevance: dict[tuple[str, str, int], str] | None = None,
) -> None:
    relevance = relevance or {}
    for channel in channels:
        for query_id in query_ids:
            path = root / f"{channel}_{query_id}_scores.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="", encoding="utf-8") as output_file:
                writer = csv.DictWriter(output_file, fieldnames=FIELDS)
                writer.writeheader()
                for rank in range(1, k + 1):
                    writer.writerow(
                        {
                            "report_id": f"{channel}_{query_id}",
                            "generated_at": "2026-06-02T20:00:00",
                            "scope": "full",
                            "query_id": query_id,
                            "input": query_id,
                            "retrieval_query": query_id,
                            "channel": channel,
                            "rank": rank,
                            "parent_asin": f"{channel}_{query_id}_{rank}",
                            "title": "item",
                            "brand": "brand",
                            "price": "",
                            "retrieval_score": "1.0",
                            "score_type": "score",
                            "human_relevance": relevance.get(
                                (channel, query_id, rank),
                                "2" if rank <= 2 else "1" if rank <= 5 else "0",
                            ),
                            "notes": "",
                        }
                    )


def test_evaluate_retrieval_audit_writes_summary_without_language_split(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "scores"
    output_csv = tmp_path / "summary.csv"
    output_md = tmp_path / "summary.md"
    write_scores(input_root)

    rows = evaluate_retrieval_audit(
        input_glob=str(input_root / "*_scores.csv"),
        output_csv=output_csv,
        output_md=output_md,
        k=10,
    )

    sections = {row["section"] for row in rows}
    assert sections == {"overall", "by_query", "pairwise_wins"}
    assert "by_language" not in sections
    assert output_csv.exists()
    assert output_md.exists()
    with output_csv.open(newline="", encoding="utf-8") as input_file:
        csv_rows = list(csv.DictReader(input_file))
    overall = [row for row in csv_rows if row["section"] == "overall"]
    assert [row["channel"] for row in overall] == ["e5", "blair", "bm25"]
    assert overall[0]["queries"] == "2"
    assert overall[0]["results"] == "20"
    assert overall[0]["mean_relevance@10"] == "0.700000"
    assert "by_language" not in output_md.read_text(encoding="utf-8")


def test_missing_score_fails(tmp_path: Path) -> None:
    input_root = tmp_path / "scores"
    write_scores(input_root, relevance={("e5", "EN-01", 1): ""})

    with pytest.raises(ValueError, match="human_relevance must not be empty"):
        load_scores(sorted(input_root.glob("*_scores.csv")))


def test_invalid_score_fails(tmp_path: Path) -> None:
    input_root = tmp_path / "scores"
    write_scores(input_root, relevance={("e5", "EN-01", 1): "3"})

    with pytest.raises(ValueError, match="human_relevance must be one of 0, 1, or 2"):
        load_scores(sorted(input_root.glob("*_scores.csv")))


def test_query_set_mismatch_fails(tmp_path: Path) -> None:
    input_root = tmp_path / "scores"
    write_scores(input_root, channels=("e5",), query_ids=("EN-01", "ZH-01"))
    write_scores(input_root, channels=("blair", "bm25"), query_ids=("EN-01",))
    scores = load_scores(sorted(input_root.glob("*_scores.csv")))

    with pytest.raises(ValueError, match="query set mismatch"):
        group_scores(scores, k=10)


def test_duplicate_rank_fails(tmp_path: Path) -> None:
    input_root = tmp_path / "scores"
    write_scores(input_root)
    path = input_root / "e5_EN-01_scores.csv"
    with path.open(newline="", encoding="utf-8") as input_file:
        rows = list(csv.DictReader(input_file))
    rows[1]["rank"] = "1"
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    scores = load_scores(sorted(input_root.glob("*_scores.csv")))

    with pytest.raises(ValueError, match="duplicate rank"):
        group_scores(scores, k=10)
