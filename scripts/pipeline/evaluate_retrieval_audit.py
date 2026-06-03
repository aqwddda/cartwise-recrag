"""Summarize phase-six retrieval audit CSV scores."""

from __future__ import annotations

import argparse
import csv
import glob
import math
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

from scripts.paths import PROJECT_ROOT


CHANNELS = ("e5", "blair", "bm25")
DEFAULT_INPUT_GLOB = "artifacts/reports/manual_testing/*_scores.csv"
DEFAULT_K = 10
DEFAULT_OUTPUT_CSV = (
    PROJECT_ROOT / "reports" / "metrics" / "full" / "retrieval_audit_summary.csv"
)
DEFAULT_OUTPUT_MD = (
    PROJECT_ROOT / "reports" / "metrics" / "full" / "retrieval_audit_summary.md"
)
REQUIRED_FIELDS = {
    "query_id",
    "channel",
    "rank",
    "parent_asin",
    "human_relevance",
}


@dataclass(frozen=True)
class AuditScore:
    query_id: str
    channel: str
    rank: int
    parent_asin: str
    relevance: int


@dataclass(frozen=True)
class MetricSummary:
    queries: int
    results: int
    mean_relevance: float
    excellent_rate: float
    acceptable_rate: float
    bad_rate: float
    precision_strict: float
    precision_loose: float
    ndcg: float
    mrr_strict: float
    mrr_loose: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glob", default=DEFAULT_INPUT_GLOB)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    return parser.parse_args()


def resolve_input_files(input_glob: str) -> list[Path]:
    pattern = Path(input_glob)
    if pattern.is_absolute():
        paths = sorted(Path(path) for path in glob.glob(input_glob))
    else:
        paths = sorted(PROJECT_ROOT.glob(input_glob))
    if not paths:
        raise ValueError(f"no audit score CSV files matched: {input_glob}")
    return paths


def load_scores(files: list[Path]) -> list[AuditScore]:
    scores: list[AuditScore] = []
    for path in files:
        with path.open(newline="", encoding="utf-8-sig") as input_file:
            reader = csv.DictReader(input_file)
            fieldnames = set(reader.fieldnames or [])
            missing_fields = REQUIRED_FIELDS - fieldnames
            if missing_fields:
                fields = ", ".join(sorted(missing_fields))
                raise ValueError(f"{path} missing required field(s): {fields}")
            for line_number, row in enumerate(reader, start=2):
                scores.append(parse_score_row(path, line_number, row))
    return scores


def parse_score_row(
    path: Path, line_number: int, row: dict[str, str]
) -> AuditScore:
    channel = row["channel"].strip().casefold()
    if channel not in CHANNELS:
        raise ValueError(f"{path}:{line_number} unsupported channel: {channel}")
    query_id = row["query_id"].strip()
    parent_asin = row["parent_asin"].strip()
    relevance_text = row["human_relevance"].strip()
    if not query_id:
        raise ValueError(f"{path}:{line_number} query_id must not be empty")
    if not parent_asin:
        raise ValueError(f"{path}:{line_number} parent_asin must not be empty")
    if relevance_text == "":
        raise ValueError(f"{path}:{line_number} human_relevance must not be empty")
    if relevance_text not in {"0", "1", "2"}:
        raise ValueError(
            f"{path}:{line_number} human_relevance must be one of 0, 1, or 2"
        )
    try:
        rank = int(row["rank"])
    except ValueError as error:
        raise ValueError(f"{path}:{line_number} rank must be an integer") from error
    if rank <= 0:
        raise ValueError(f"{path}:{line_number} rank must be greater than zero")
    return AuditScore(
        query_id=query_id,
        channel=channel,
        rank=rank,
        parent_asin=parent_asin,
        relevance=int(relevance_text),
    )


def group_scores(
    scores: list[AuditScore], *, k: int
) -> dict[str, dict[str, list[AuditScore]]]:
    if k <= 0:
        raise ValueError("--k must be greater than zero")
    grouped: dict[str, dict[str, list[AuditScore]]] = {
        channel: defaultdict(list) for channel in CHANNELS
    }
    seen_ranks: set[tuple[str, str, int]] = set()
    seen_items: set[tuple[str, str, str]] = set()
    for score in scores:
        if score.rank > k:
            continue
        rank_key = (score.channel, score.query_id, score.rank)
        item_key = (score.channel, score.query_id, score.parent_asin)
        if rank_key in seen_ranks:
            raise ValueError(
                f"duplicate rank for {score.channel} {score.query_id}: {score.rank}"
            )
        if item_key in seen_items:
            raise ValueError(
                f"duplicate item for {score.channel} {score.query_id}: "
                f"{score.parent_asin}"
            )
        seen_ranks.add(rank_key)
        seen_items.add(item_key)
        grouped[score.channel][score.query_id].append(score)

    query_sets = {channel: set(grouped[channel]) for channel in CHANNELS}
    reference_queries = query_sets[CHANNELS[0]]
    for channel in CHANNELS[1:]:
        if query_sets[channel] != reference_queries:
            missing = sorted(reference_queries - query_sets[channel])
            extra = sorted(query_sets[channel] - reference_queries)
            raise ValueError(
                f"query set mismatch for {channel}: missing={missing}, extra={extra}"
            )

    for channel in CHANNELS:
        for query_id, query_scores in grouped[channel].items():
            if len(query_scores) != k:
                raise ValueError(
                    f"{channel} {query_id} has {len(query_scores)} scored rows; "
                    f"expected {k}"
                )
            query_scores.sort(key=lambda score: score.rank)
            expected_ranks = list(range(1, k + 1))
            actual_ranks = [score.rank for score in query_scores]
            if actual_ranks != expected_ranks:
                raise ValueError(
                    f"{channel} {query_id} ranks must be contiguous 1..{k}; "
                    f"got {actual_ranks}"
                )
    return {channel: dict(queries) for channel, queries in grouped.items()}


def summarize(query_scores: list[list[AuditScore]], *, k: int) -> MetricSummary:
    if not query_scores:
        return MetricSummary(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    flat = [score.relevance for scores in query_scores for score in scores]
    results = len(flat)
    excellent = sum(1 for relevance in flat if relevance == 2)
    acceptable = sum(1 for relevance in flat if relevance >= 1)
    bad = sum(1 for relevance in flat if relevance == 0)
    return MetricSummary(
        queries=len(query_scores),
        results=results,
        mean_relevance=sum(flat) / results,
        excellent_rate=excellent / results,
        acceptable_rate=acceptable / results,
        bad_rate=bad / results,
        precision_strict=excellent / results,
        precision_loose=acceptable / results,
        ndcg=sum(ndcg_at_k(scores, k=k) for scores in query_scores) / len(query_scores),
        mrr_strict=sum(mrr_at_k(scores, threshold=2, k=k) for scores in query_scores)
        / len(query_scores),
        mrr_loose=sum(mrr_at_k(scores, threshold=1, k=k) for scores in query_scores)
        / len(query_scores),
    )


def dcg(relevances: list[int], *, k: int) -> float:
    return sum(
        ((2**relevance) - 1) / math.log2(index + 2)
        for index, relevance in enumerate(relevances[:k])
    )


def ndcg_at_k(scores: list[AuditScore], *, k: int) -> float:
    relevances = [score.relevance for score in scores[:k]]
    ideal = sorted(relevances, reverse=True)
    ideal_dcg = dcg(ideal, k=k)
    if ideal_dcg == 0:
        return 0.0
    return dcg(relevances, k=k) / ideal_dcg


def mrr_at_k(scores: list[AuditScore], *, threshold: int, k: int) -> float:
    for score in scores[:k]:
        if score.relevance >= threshold:
            return 1 / score.rank
    return 0.0


def metric_rows(
    grouped: dict[str, dict[str, list[AuditScore]]], *, k: int
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for channel in CHANNELS:
        summary = summarize(list(grouped[channel].values()), k=k)
        rows.append(format_metric_row("overall", channel, "", summary, k=k))

    query_ids = sorted(next(iter(grouped.values())))
    for query_id in query_ids:
        for channel in CHANNELS:
            summary = summarize([grouped[channel][query_id]], k=k)
            rows.append(format_metric_row("by_query", channel, query_id, summary, k=k))

    rows.extend(pairwise_rows(grouped, k=k))
    return rows


def format_metric_row(
    section: str,
    channel: str,
    query_id: str,
    summary: MetricSummary,
    *,
    k: int,
    compared_model: str = "",
    wins: int | str = "",
    losses: int | str = "",
    ties: int | str = "",
) -> dict[str, str]:
    return {
        "section": section,
        "channel": channel,
        "compared_model": str(compared_model),
        "query_id": query_id,
        "queries": str(summary.queries),
        "results": str(summary.results),
        f"mean_relevance@{k}": format_float(summary.mean_relevance),
        f"excellent_rate@{k}": format_float(summary.excellent_rate),
        f"acceptable_rate@{k}": format_float(summary.acceptable_rate),
        f"bad_rate@{k}": format_float(summary.bad_rate),
        f"precision_strict@{k}": format_float(summary.precision_strict),
        f"precision_loose@{k}": format_float(summary.precision_loose),
        f"ndcg@{k}": format_float(summary.ndcg),
        f"mrr_strict@{k}": format_float(summary.mrr_strict),
        f"mrr_loose@{k}": format_float(summary.mrr_loose),
        "wins": str(wins),
        "losses": str(losses),
        "ties": str(ties),
    }


def pairwise_rows(
    grouped: dict[str, dict[str, list[AuditScore]]], *, k: int
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    query_ids = sorted(next(iter(grouped.values())))
    for left, right in combinations(CHANNELS, 2):
        left_wins = right_wins = ties = 0
        for query_id in query_ids:
            left_summary = summarize([grouped[left][query_id]], k=k)
            right_summary = summarize([grouped[right][query_id]], k=k)
            comparison = compare_summaries(left_summary, right_summary)
            if comparison > 0:
                left_wins += 1
            elif comparison < 0:
                right_wins += 1
            else:
                ties += 1
        empty = MetricSummary(len(query_ids), 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        rows.append(
            format_metric_row(
                "pairwise_wins",
                left,
                "",
                empty,
                k=k,
                compared_model=right,
                wins=left_wins,
                losses=right_wins,
                ties=ties,
            )
        )
        rows.append(
            format_metric_row(
                "pairwise_wins",
                right,
                "",
                empty,
                k=k,
                compared_model=left,
                wins=right_wins,
                losses=left_wins,
                ties=ties,
            )
        )
    return rows


def compare_summaries(left: MetricSummary, right: MetricSummary) -> int:
    left_key = (left.ndcg, left.mean_relevance)
    right_key = (right.ndcg, right.mean_relevance)
    if left_key > right_key:
        return 1
    if left_key < right_key:
        return -1
    return 0


def format_float(value: float) -> str:
    return f"{value:.6f}"


def write_csv(output: Path, rows: list[dict[str, str]], *, k: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "section",
        "channel",
        "compared_model",
        "query_id",
        "queries",
        "results",
        f"mean_relevance@{k}",
        f"excellent_rate@{k}",
        f"acceptable_rate@{k}",
        f"bad_rate@{k}",
        f"precision_strict@{k}",
        f"precision_loose@{k}",
        f"ndcg@{k}",
        f"mrr_strict@{k}",
        f"mrr_loose@{k}",
        "wins",
        "losses",
        "ties",
    ]
    partial = output.with_suffix(f"{output.suffix}.part")
    with partial.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    partial.replace(output)


def write_markdown(output: Path, rows: list[dict[str, str]], *, k: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    overall = [row for row in rows if row["section"] == "overall"]
    pairwise = [row for row in rows if row["section"] == "pairwise_wins"]
    metric_names = [
        f"mean_relevance@{k}",
        f"excellent_rate@{k}",
        f"acceptable_rate@{k}",
        f"bad_rate@{k}",
        f"ndcg@{k}",
        f"mrr_strict@{k}",
        f"mrr_loose@{k}",
    ]
    lines = [
        "# Phase 6 Retrieval Audit Summary",
        "",
        "## Overall",
        "",
        markdown_table(["channel", "queries", "results", *metric_names], overall),
        "",
        "## Pairwise Wins",
        "",
        markdown_table(
            ["channel", "compared_model", "queries", "wins", "losses", "ties"],
            pairwise,
        ),
        "",
        "## Notes",
        "",
        "- Scores use the human relevance labels 0, 1, and 2.",
        f"- Pairwise wins compare per-query `ndcg@{k}`, then `mean_relevance@{k}`.",
        "- This summary intentionally does not split results by query language.",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")


def markdown_table(columns: list[str], rows: list[dict[str, str]]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(row.get(column, "") for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def evaluate_retrieval_audit(
    *,
    input_glob: str = DEFAULT_INPUT_GLOB,
    output_csv: Path = DEFAULT_OUTPUT_CSV,
    output_md: Path = DEFAULT_OUTPUT_MD,
    k: int = DEFAULT_K,
) -> list[dict[str, str]]:
    files = resolve_input_files(input_glob)
    scores = load_scores(files)
    grouped = group_scores(scores, k=k)
    rows = metric_rows(grouped, k=k)
    write_csv(output_csv, rows, k=k)
    write_markdown(output_md, rows, k=k)
    print(f"Wrote retrieval audit summary CSV: {output_csv}")
    print(f"Wrote retrieval audit summary Markdown: {output_md}")
    return rows


def main() -> None:
    args = parse_args()
    evaluate_retrieval_audit(
        input_glob=args.input_glob,
        output_csv=args.output_csv,
        output_md=args.output_md,
        k=args.k,
    )


if __name__ == "__main__":
    main()
