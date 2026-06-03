from __future__ import annotations

from scripts.tools.generate_brand_stats import build_brand_records


def test_build_brand_records_groups_normalized_brands_and_counts_missing() -> None:
    records, summary = build_brand_records(
        [
            {"parent_asin": "P1", "brand": " Fender "},
            {"parent_asin": "P2", "brand": "fender"},
            {"parent_asin": "P3", "brand": "Shure"},
            {"parent_asin": "P4", "brand": ""},
            {"parent_asin": "P5", "brand": None},
        ]
    )

    assert summary == {
        "item_count": 5,
        "branded_item_count": 3,
        "missing_brand_count": 2,
        "unique_brand_count": 2,
    }
    assert [(record.brand, record.item_count) for record in records] == [
        ("Fender", 2),
        ("Shure", 1),
    ]
    assert records[0].example_parent_asin == "P1"
