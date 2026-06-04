"""Build a deterministic small dataset for local development."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.paths import DEV_PROCESSED_ROOT
from scripts.pipeline.preprocess_amazon_reviews import (
    DEFAULT_RAW_ROOT,
    DEFAULT_MAX_LOW_RATING_REVIEWS_PER_ITEM,
    DEFAULT_MAX_REVIEWS_PER_ITEM,
    run_preprocessing,
)


DEFAULT_OUTPUT_ROOT = DEV_PROCESSED_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-items", type=int, default=500)
    parser.add_argument(
        "--max-reviews-per-item", type=int, default=DEFAULT_MAX_REVIEWS_PER_ITEM
    )
    parser.add_argument(
        "--max-low-rating-reviews-per-item",
        type=int,
        default=DEFAULT_MAX_LOW_RATING_REVIEWS_PER_ITEM,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_preprocessing(
        raw_root=args.raw_root,
        output_root=args.output_root,
        max_items=args.max_items,
        max_reviews_per_item=args.max_reviews_per_item,
        max_low_rating_reviews_per_item=args.max_low_rating_reviews_per_item,
    )


if __name__ == "__main__":
    main()
