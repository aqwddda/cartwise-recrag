"""Recall popular products and manually experiment with hard filters."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from cartwise.retrieval.filters import (
    FilterConstraints,
    apply_hard_filters,
    derive_category_tags,
    derive_color_tags,
    derive_material_tags,
)
from cartwise.retrieval.popularity import PopularityRecommender
from scripts.paths import PROCESSED_ROOT
from scripts.tools.item_metadata import load_items_by_parent_asin

DEFAULT_PROCESSED_ROOT = PROCESSED_ROOT
DEFAULT_USER_ID = "manual-filter-demo-user"
RECALL_COUNT = 50


def configure_console_output() -> None:
    """Keep metadata previews printable in terminals with limited encodings."""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")


def build_constraints() -> FilterConstraints:
    """Edit this function to try different hard-filter combinations."""

    return FilterConstraints(
        # Empty collections mean that the corresponding filter is disabled.
        # Try: category_tags={"guitar", "tuner"}
        # Try: excluded_brands={"SNARK"}
        # Try: color_tags={"black"}, material_tags={"copper"}
        category_tags=(),
        min_price=10.0,
        max_price=50.0,
        brands=(),
        excluded_brands=(),
        color_tags={"black"},
        material_tags=(),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--user-id", default=DEFAULT_USER_ID)
    return parser.parse_args()


def format_tags(tags: set[str]) -> str:
    return ", ".join(sorted(tags)) or "-"


def print_items(title: str, items: list[dict[str, Any]]) -> None:
    print(f"\n{title}: {len(items)} item(s)")
    for rank, item in enumerate(items, start=1):
        price = item["price"]
        price_text = f"${price:.2f}" if price is not None else "missing"
        print(
            f"{rank:>2}. {item['parent_asin']} | {price_text} | "
            f"brand={item['brand'] or '-'}"
        )
        print(f"    category_tags={format_tags(derive_category_tags(item))}")
        print(f"    color_tags={format_tags(derive_color_tags(item))}")
        print(f"    material_tags={format_tags(derive_material_tags(item))}")
        print(f"    title={item['title'] or '-'}")


def main() -> None:
    configure_console_output()
    args = parse_args()
    training_path = args.processed_root / "interactions_train.parquet"
    items_path = args.processed_root / "items.parquet"

    recommender = PopularityRecommender.from_parquet(training_path)
    recalled_parent_asins = recommender.recommend(args.user_id, k=RECALL_COUNT)
    items_by_parent_asin = load_items_by_parent_asin(
        items_path,
        columns=[
            "parent_asin",
            "title",
            "brand",
            "price",
            "categories",
            "details_json",
        ],
    )
    recalled_items = [
        items_by_parent_asin[parent_asin]
        for parent_asin in recalled_parent_asins
        if parent_asin in items_by_parent_asin
    ]

    constraints = build_constraints()
    filtered_items = apply_hard_filters(recalled_items, constraints)

    print(f"User ID: {args.user_id}")
    print(f"Constraints: {constraints}")
    # print_items("Popularity recalled candidates", recalled_items)
    print_items("Candidates after hard filters", filtered_items)


if __name__ == "__main__":
    main()
