from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cartwise.retrieval.filters import (
    FilterConstraints,
    apply_hard_filters,
    derive_category_tags,
    load_brand_alias_to_canonical,
    load_item_to_categories,
    resolve_filter_constraints,
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


def test_derive_category_tags_uses_all_product_categories() -> None:
    title_only = make_item("P1", title="Clip-On Guitar Tuner")
    categorized = make_item(
        "P2",
        categories=["Musical Instruments", "Microphones & Accessories", "Stands"],
    )

    assert derive_category_tags(title_only) == set()
    assert derive_category_tags(categorized) == {
        "musical instruments",
        "microphones & accessories",
        "stands",
    }


def test_resolve_filter_constraints_maps_llm_terms_through_offline_tables() -> None:
    constraints = resolve_filter_constraints(
        product_terms=["electric guitar", "unknown product"],
        brands=["fender", "unknown brand"],
        excluded_brands=["not shure"],
        min_price=10,
        max_price=20,
        color_tags=["black"],
        material_tags=["wood"],
        item_to_categories={"electric guitar": "Guitars"},
        brand_alias_to_canonical={"fender": "Fender", "not shure": "Shure"},
    )

    assert tuple(constraints.category_tags) == ("Guitars",)
    assert constraints.min_price == 10
    assert constraints.max_price == 20
    assert tuple(constraints.brands) == ("Fender",)
    assert tuple(constraints.excluded_brands) == ("Shure",)
    assert tuple(constraints.color_tags) == ("black",)
    assert tuple(constraints.material_tags) == ("wood",)


def test_offline_mapping_loaders_normalize_keys_and_keep_canonical_values(
    tmp_path: Path,
) -> None:
    item_path = tmp_path / "item_to_categories.json"
    brand_path = tmp_path / "brand_alias_to_canonical.json"
    item_path.write_text(
        json.dumps({" Electric Guitar ": "Guitars", "bad": ""}),
        encoding="utf-8",
    )
    brand_path.write_text(
        json.dumps({" FENDER ": "Fender", "bad": ""}),
        encoding="utf-8",
    )

    assert load_item_to_categories(item_path) == {"electric guitar": "Guitars"}
    assert load_brand_alias_to_canonical(brand_path) == {"fender": "Fender"}


def test_resolve_filter_constraints_allows_explicit_empty_mapping() -> None:
    constraints = resolve_filter_constraints(
        product_terms=["electric guitar"],
        brands=["fender"],
        item_to_categories={},
        brand_alias_to_canonical={},
    )

    assert tuple(constraints.category_tags) == ()
    assert tuple(constraints.brands) == ()


def test_category_constraint_matches_any_category_by_substring() -> None:
    candidates = [
        make_item(
            "MATCH",
            categories=[
                "Musical Instruments",
                "Instrument Accessories",
                "General Accessories",
            ],
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
        FilterConstraints(category_tags={" Accessories "}),
    )

    assert [item["parent_asin"] for item in filtered] == ["MATCH"]


def test_category_constraint_rejects_when_no_category_contains_tag() -> None:
    candidates = [
        make_item(
            "OTHER",
            categories=["Musical Instruments", "Instrument Accessories", "Tools"],
        ),
    ]

    filtered = apply_hard_filters(
        candidates,
        FilterConstraints(category_tags={"stands"}),
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
