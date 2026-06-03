"""Generate offline brand vocabulary and frequency statistics."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from cartwise.retrieval.filters import normalize_string
from scripts.paths import ARTIFACT_REPORTS_ROOT, PROCESSED_ROOT


DEFAULT_INPUT = PROCESSED_ROOT / "items.parquet"
DEFAULT_OUTPUT_ROOT = ARTIFACT_REPORTS_ROOT / "category_stats"


@dataclass(frozen=True, slots=True)
class BrandRecord:
    brand: str
    normalized_brand: str
    item_count: int
    item_share: float
    example_parent_asin: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def _display_brand(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    brand = value.strip()
    return brand or None


def build_brand_records(rows: list[dict[str, Any]]) -> tuple[list[BrandRecord], dict[str, int]]:
    item_count = len(rows)
    counts: Counter[str] = Counter()
    display_by_brand: dict[str, str] = {}
    example_by_brand: dict[str, str] = {}
    missing_brand_count = 0

    for row in rows:
        brand = _display_brand(row.get("brand"))
        normalized_brand = normalize_string(brand)
        if normalized_brand is None:
            missing_brand_count += 1
            continue
        counts[normalized_brand] += 1
        display_by_brand.setdefault(normalized_brand, brand or normalized_brand)
        example_by_brand.setdefault(normalized_brand, str(row.get("parent_asin") or ""))

    records = [
        BrandRecord(
            brand=display_by_brand[normalized_brand],
            normalized_brand=normalized_brand,
            item_count=count,
            item_share=count / item_count if item_count else 0.0,
            example_parent_asin=example_by_brand[normalized_brand],
        )
        for normalized_brand, count in counts.items()
    ]
    records.sort(key=lambda record: (-record.item_count, record.normalized_brand))
    summary = {
        "item_count": item_count,
        "branded_item_count": item_count - missing_brand_count,
        "missing_brand_count": missing_brand_count,
        "unique_brand_count": len(records),
    }
    return records, summary


def write_brand_values_csv(path: Path, records: list[BrandRecord]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "brand",
                "normalized_brand",
                "item_count",
                "item_share",
                "example_parent_asin",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "brand": record.brand,
                    "normalized_brand": record.normalized_brand,
                    "item_count": record.item_count,
                    "item_share": f"{record.item_share:.6f}",
                    "example_parent_asin": record.example_parent_asin,
                }
            )


def write_brand_vocabulary_json(
    path: Path,
    *,
    source: Path,
    records: list[BrandRecord],
    summary: dict[str, int],
) -> None:
    payload = {
        "source": str(source),
        **summary,
        "brands": [
            {
                "brand": record.brand,
                "normalized_brand": record.normalized_brand,
                "item_count": record.item_count,
                "example_parent_asin": record.example_parent_asin,
            }
            for record in records
        ],
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_brand_vocabulary_text(path: Path, records: list[BrandRecord]) -> None:
    path.write_text(
        "\n".join(record.brand for record in records) + "\n",
        encoding="utf-8",
    )


def generate_brand_stats(input_path: Path, output_root: Path) -> dict[str, int | str]:
    rows = pq.read_table(input_path, columns=["parent_asin", "brand"]).to_pylist()
    records, summary = build_brand_records(rows)
    output_root.mkdir(parents=True, exist_ok=True)

    values_path = output_root / "brand_values.csv"
    vocabulary_json_path = output_root / "brand_vocabulary.json"
    vocabulary_text_path = output_root / "brand_vocabulary.txt"

    write_brand_values_csv(values_path, records)
    write_brand_vocabulary_json(
        vocabulary_json_path,
        source=input_path,
        records=records,
        summary=summary,
    )
    write_brand_vocabulary_text(vocabulary_text_path, records)

    return {
        **summary,
        "brand_values_csv": str(values_path),
        "brand_vocabulary_json": str(vocabulary_json_path),
        "brand_vocabulary_txt": str(vocabulary_text_path),
    }


def main() -> None:
    args = parse_args()
    result = generate_brand_stats(args.input, args.output_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
