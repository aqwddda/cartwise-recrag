from __future__ import annotations

import json
from typing import Any

import pytest

from cartwise.retrieval.filters import (
    FilterConstraints,
    apply_hard_filters,
    derive_category_tags,
)


def make_item(
    parent_asin: str,
    *,
    title: str | None = None,
    brand: str | None = None,
    price: float | None = None,
    categories: list[str] | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "parent_asin": parent_asin,
        "title": title,
        "brand": brand,
        "price": price,
        "categories": categories or [],
        "details_json": json.dumps(details or {}),
    }


@pytest.fixture(autouse=True)
def allowed_leaf_categories(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "cartwise.retrieval.filters.load_allowed_leaf_categories",
        lambda: frozenset({"stands", "instrument cables"}),
    )


def test_derive_category_tags_uses_allowed_leaf_category_only() -> None:
    title_only = make_item("P1", title="Clip-On Guitar Tuner")
    allowed_leaf = make_item(
        "P2",
        categories=["Musical Instruments", "Microphones & Accessories", "Stands"],
    )
    disallowed_leaf = make_item(
        "P3",
        categories=["Musical Instruments", "Instrument Accessories", "Tools"],
    )

    assert derive_category_tags(title_only) == set()
    assert derive_category_tags(allowed_leaf) == {"stands"}
    assert derive_category_tags(disallowed_leaf) == set()


def test_category_constraint_excludes_items_without_matching_tags() -> None:
    candidates = [
        make_item(
            "MATCH",
            categories=["Musical Instruments", "Microphones & Accessories", "Stands"],
        ),
        make_item("MISSING"),
        make_item(
            "OTHER",
            categories=[
                "Musical Instruments",
                "Live Sound & Stage",
                "Instrument Cables",
            ],
        ),
    ]

    filtered = apply_hard_filters(
        candidates,
        FilterConstraints(category_tags={" STANDS "}),
    )

    assert [item["parent_asin"] for item in filtered] == ["MATCH"]


def test_category_constraint_rejects_leaf_categories_outside_stage_seven_table() -> None:
    candidates = [
        make_item(
            "DISALLOWED",
            categories=["Musical Instruments", "Instrument Accessories", "Tools"],
        ),
    ]

    filtered = apply_hard_filters(
        candidates,
        FilterConstraints(category_tags={"tools"}),
    )

    assert filtered == []


def test_missing_category_is_allowed_without_category_constraint() -> None:
    candidates = [make_item("MISSING")]

    assert apply_hard_filters(candidates, FilterConstraints()) == candidates


def test_price_range_includes_both_boundaries() -> None:
    candidates = [
        make_item("LOW", price=10.0),
        make_item("MIDDLE", price=15.0),
        make_item("HIGH", price=20.0),
        make_item("BELOW", price=9.99),
        make_item("ABOVE", price=20.01),
    ]

    filtered = apply_hard_filters(
        candidates,
        FilterConstraints(min_price=10.0, max_price=20.0),
    )

    assert [item["parent_asin"] for item in filtered] == [
        "LOW",
        "MIDDLE",
        "HIGH",
    ]


def test_missing_price_is_excluded_only_when_price_is_constrained() -> None:
    candidates = [make_item("MISSING"), make_item("KNOWN", price=20.0)]

    assert apply_hard_filters(candidates, FilterConstraints()) == candidates
    assert apply_hard_filters(
        candidates,
        FilterConstraints(max_price=25.0),
    ) == [candidates[1]]


def test_brand_color_and_material_comparisons_normalize_whitespace_and_case() -> None:
    candidates = [
        make_item(
            "MATCH",
            brand="  Fender ",
            details={"Color Name": " BLACK ", "Material Type": " Wood "},
        ),
        make_item(
            "EXCLUDED_BRAND",
            brand=" GIBSON ",
            details={"Color": "black", "Material": "wood"},
        ),
        make_item(
            "WRONG_COLOR",
            brand="FENDER",
            details={"Color": "white", "Material": "WOOD"},
        ),
    ]

    filtered = apply_hard_filters(
        candidates,
        FilterConstraints(
            brands={" fEnDeR  "},
            excluded_brands={" gibson "},
            color_tags={" black "},
            material_tags={" WOOD "},
        ),
    )

    assert [item["parent_asin"] for item in filtered] == ["MATCH"]


def test_missing_color_or_material_is_excluded_only_when_constrained() -> None:
    candidates = [
        make_item("MISSING"),
        make_item("COLOR_ONLY", details={"Color": "black"}),
        make_item("MATERIAL_ONLY", details={"Material": "wood"}),
        make_item("BOTH", details={"Color": "black", "Material": "wood"}),
    ]

    assert apply_hard_filters(candidates, FilterConstraints()) == candidates
    assert apply_hard_filters(
        candidates,
        FilterConstraints(color_tags={"black"}, material_tags={"wood"}),
    ) == [candidates[3]]


def test_excluded_parent_asins_is_reserved_but_not_applied_in_stage_four() -> None:
    candidates = [make_item("P1"), make_item("P2")]

    assert apply_hard_filters(
        candidates,
        FilterConstraints(),
        excluded_parent_asins={"P1"},
    ) == candidates


def test_filter_preserves_order_and_does_not_pad_short_results() -> None:
    candidates = [
        make_item("P1", price=40.0),
        make_item("P2", price=10.0),
        make_item("P3", price=30.0),
        make_item("P4", price=20.0),
    ]

    filtered = apply_hard_filters(
        candidates,
        FilterConstraints(max_price=30.0),
    )

    assert [item["parent_asin"] for item in filtered] == ["P2", "P3", "P4"]
