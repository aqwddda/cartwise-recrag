"""Evaluate the Popularity recommendation baseline on chronological splits."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from cartwise.retrieval.popularity import (
    PopularityRecommender,
    RankingMetrics,
    evaluate_recommender,
    load_interactions,
)
from scripts.paths import METRICS_ROOT, PROCESSED_ROOTS, PROJECT_ROOT


DEFAULT_SCOPE = "dev"
SCOPE_PATHS = {
    "dev": (
        PROCESSED_ROOTS["dev"],
        METRICS_ROOT / "dev" / "popularity.csv",
    ),
    "full": (
        PROCESSED_ROOTS["full"],
        METRICS_ROOT / "full" / "popularity.csv",
    ),
}
DEFAULT_PROCESSED_ROOT, DEFAULT_OUTPUT = SCOPE_PATHS[DEFAULT_SCOPE]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=SCOPE_PATHS, default=DEFAULT_SCOPE)
    parser.add_argument("--processed-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--k", type=int, default=10)
    return parser.parse_args()


def resolve_paths(
    scope: str,
    processed_root: Path | None = None,
    output: Path | None = None,
) -> tuple[Path, Path]:
    default_processed_root, default_output = SCOPE_PATHS[scope]
    return processed_root or default_processed_root, output or default_output


def metrics_row(
    split: str, metrics: RankingMetrics, *, k: int
) -> dict[str, str | int]:
    return format_metrics_row("popularity", split, metrics, k=k)


def format_metrics_row(
    model: str, split: str, metrics: RankingMetrics, *, k: int
) -> dict[str, str | int]:
    """Format metrics consistently across recommendation models."""

    return {
        "model": model,
        "split": split,
        "users": metrics.users,
        f"Recall@{k}": f"{metrics.recall:.6f}",
        f"NDCG@{k}": f"{metrics.ndcg:.6f}",
        f"HitRate@{k}": f"{metrics.hit_rate:.6f}",
    }


def write_metrics_csv(
    output: Path,
    rows: list[dict[str, str | int]],
    *,
    k: int,
    additional_ks: list[int] | None = None,
    extra_fieldnames: list[str] | None = None,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(f"{output.suffix}.part")
    with partial.open("w", newline="", encoding="utf-8") as output_file:
        fieldnames = ["model", "split", "users"]
        for metric_k in dict.fromkeys([*(additional_ks or []), k]):
            fieldnames.extend(
                [
                    f"Recall@{metric_k}",
                    f"NDCG@{metric_k}",
                    f"HitRate@{metric_k}",
                ]
            )
        fieldnames.extend(extra_fieldnames or [])
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    partial.replace(output)


def evaluate_popularity(
    processed_root: Path = DEFAULT_PROCESSED_ROOT,
    output: Path = DEFAULT_OUTPUT,
    *,
    k: int = 10,
) -> list[dict[str, str | int]]:
    train = load_interactions(processed_root / "interactions_train.parquet")
    valid = load_interactions(processed_root / "interactions_valid.parquet")
    test = load_interactions(processed_root / "interactions_test.parquet")
    recommender = PopularityRecommender(train)

    rows = [
        metrics_row("valid", evaluate_recommender(recommender, valid, k=k), k=k),
        metrics_row(
            "test",
            evaluate_recommender(
                recommender,
                test,
                k=k,
                additional_history=valid,
            ),
            k=k,
        ),
    ]
    write_metrics_csv(output, rows, k=k)
    print(f"Wrote metrics: {output}")
    for row in rows:
        print(row)
    return rows


def main() -> None:
    args = parse_args()
    processed_root, output = resolve_paths(
        args.scope,
        processed_root=args.processed_root,
        output=args.output,
    )
    evaluate_popularity(processed_root, output, k=args.k)


if __name__ == "__main__":
    main()
