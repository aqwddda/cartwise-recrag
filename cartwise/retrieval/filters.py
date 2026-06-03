"""Metadata-derived hard filters for ranked product candidates."""

from __future__ import annotations

import json
import math
from functools import lru_cache
from collections.abc import Collection, Iterable, Iterator, Mapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, TypeVar


Candidate = TypeVar("Candidate", bound=Mapping[str, Any])

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ITEM_TO_CATEGORIES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "item_to_categories.json"
)
DEFAULT_BRAND_ALIAS_TO_CANONICAL_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "brand_alias_to_canonical.json"
)


@dataclass(frozen=True, slots=True)
class FilterConstraints:
    """Explicit user constraints applied after candidate ranking."""

    category_tags: Collection[str] = ()
    min_price: float | None = None
    max_price: float | None = None
    brands: Collection[str] = ()
    excluded_brands: Collection[str] = ()
    color_tags: Collection[str] = ()
    material_tags: Collection[str] = ()


@dataclass(frozen=True, slots=True)
class _NormalizedConstraints:
    category_tags: frozenset[str]
    min_price: float | None
    max_price: float | None
    brands: frozenset[str]
    excluded_brands: frozenset[str]
    color_tags: frozenset[str]
    material_tags: frozenset[str]


def normalize_string(value: Any) -> str | None:
    """Normalize text used for exact metadata comparisons."""

    if value is None:
        return None
    normalized = str(value).strip().casefold()
    return normalized or None


def _normalize_strings(values: Collection[str]) -> frozenset[str]:
    return frozenset(
        normalized
        for value in values
        if (normalized := normalize_string(value)) is not None
    )


def _load_json_mapping(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, Mapping) else {}


@lru_cache(maxsize=8)
def load_item_to_categories(path: Path | None = None) -> dict[str, str]:
    """Load product-term aliases mapped to canonical category tags."""

    source = path or DEFAULT_ITEM_TO_CATEGORIES_PATH
    return {
        normalized_key: str(value).strip()
        for key, value in _load_json_mapping(source).items()
        if (normalized_key := normalize_string(key)) is not None
        and isinstance(value, str)
        and value.strip()
    }


@lru_cache(maxsize=8)
def load_brand_alias_to_canonical(path: Path | None = None) -> dict[str, str]:
    """Load brand aliases mapped to canonical product brands."""

    source = path or DEFAULT_BRAND_ALIAS_TO_CANONICAL_PATH
    return {
        normalized_key: str(value).strip()
        for key, value in _load_json_mapping(source).items()
        if (normalized_key := normalize_string(key)) is not None
        and isinstance(value, str)
        and value.strip()
    }


def _resolve_aliases(
    values: Collection[str],
    alias_to_canonical: Mapping[str, str],
) -> tuple[str, ...]:
    resolved: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = normalize_string(value)
        if normalized is None:
            continue
        canonical = alias_to_canonical.get(normalized)
        canonical_normalized = normalize_string(canonical)
        if canonical is None or canonical_normalized is None:
            continue
        if canonical_normalized in seen:
            continue
        resolved.append(canonical)
        seen.add(canonical_normalized)
    return tuple(resolved)


def resolve_filter_constraints(
    *,
    product_terms: Collection[str] = (),
    brands: Collection[str] = (),
    excluded_brands: Collection[str] = (),
    min_price: float | None = None,
    max_price: float | None = None,
    color_tags: Collection[str] = (),
    material_tags: Collection[str] = (),
    item_to_categories: Mapping[str, str] | None = None,
    brand_alias_to_canonical: Mapping[str, str] | None = None,
) -> FilterConstraints:
    """Map raw LLM intent fields to final filter constraints."""

    category_aliases = (
        load_item_to_categories()
        if item_to_categories is None
        else item_to_categories
    )
    brand_aliases = (
        load_brand_alias_to_canonical()
        if brand_alias_to_canonical is None
        else brand_alias_to_canonical
    )
    return FilterConstraints(
        category_tags=_resolve_aliases(product_terms, category_aliases),
        min_price=min_price,
        max_price=max_price,
        brands=_resolve_aliases(brands, brand_aliases),
        excluded_brands=_resolve_aliases(excluded_brands, brand_aliases),
        color_tags=color_tags,
        material_tags=material_tags,
    )


def _normalize_price_bound(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be a finite number")
    return normalized


def _normalize_constraints(constraints: FilterConstraints) -> _NormalizedConstraints:
    min_price = _normalize_price_bound(constraints.min_price, "min_price")
    max_price = _normalize_price_bound(constraints.max_price, "max_price")
    if min_price is not None and max_price is not None and min_price > max_price:
        raise ValueError("min_price must not exceed max_price")
    return _NormalizedConstraints(
        category_tags=_normalize_strings(constraints.category_tags),
        min_price=min_price,
        max_price=max_price,
        brands=_normalize_strings(constraints.brands),
        excluded_brands=_normalize_strings(constraints.excluded_brands),
        color_tags=_normalize_strings(constraints.color_tags),
        material_tags=_normalize_strings(constraints.material_tags),
    )


def _load_details(item: Mapping[str, Any]) -> Mapping[str, Any]:
    details = item.get("details_json")
    if isinstance(details, Mapping):
        return details
    if not isinstance(details, str):
        return {}
    try:
        parsed = json.loads(details)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, Mapping) else {}


def _iter_text_values(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        if normalized := normalize_string(value):
            yield normalized
        return
    if isinstance(value, Collection) and not isinstance(value, Mapping):
        for item in value:
            yield from _iter_text_values(item)


def derive_category_tags(item: Mapping[str, Any]) -> set[str]:
    """Derive normalized category strings from all product categories."""

    categories = item.get("categories")
    if not isinstance(categories, Collection) or isinstance(
        categories,
        (str, bytes, Mapping),
    ):
        return set()
    return {
        normalized
        for value in categories
        if (normalized := normalize_string(value)) is not None
    }


def _derive_detail_tags(item: Mapping[str, Any], keys: Iterable[str]) -> set[str]:
    details = _load_details(item)
    return {
        text
        for key in keys
        for text in _iter_text_values(details.get(key))
    }


def derive_color_tags(item: Mapping[str, Any]) -> set[str]:
    """Merge normalized color values from supported metadata fields."""

    return _derive_detail_tags(item, ("Color Name", "Color"))


def derive_material_tags(item: Mapping[str, Any]) -> set[str]:
    """Merge normalized material values from supported metadata fields."""

    return _derive_detail_tags(item, ("Material Type", "Material"))


def _read_price(item: Mapping[str, Any]) -> float | None:
    value = item.get("price")
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    price = float(value)
    return price if math.isfinite(price) else None


def _matches_category_constraints(
    item: Mapping[str, Any],
    category_tags: frozenset[str],
) -> bool:
    if not category_tags:
        return True
    item_categories = derive_category_tags(item)
    if not item_categories:
        return False
    return any(
        category_tag in item_category
        for category_tag in category_tags
        for item_category in item_categories
    )


def _matches_constraints(
    item: Mapping[str, Any],
    constraints: _NormalizedConstraints,
) -> bool:
    if not _matches_category_constraints(item, constraints.category_tags):
        return False

    has_price_constraint = (
        constraints.min_price is not None or constraints.max_price is not None
    )
    price = _read_price(item)
    if has_price_constraint and price is None:
        return False
    if constraints.min_price is not None and price < constraints.min_price:
        return False
    if constraints.max_price is not None and price > constraints.max_price:
        return False

    brand = normalize_string(item.get("brand"))
    if constraints.brands and brand not in constraints.brands:
        return False
    if brand in constraints.excluded_brands:
        return False

    if constraints.color_tags and not constraints.color_tags.intersection(
        derive_color_tags(item)
    ):
        return False
    if constraints.material_tags and not constraints.material_tags.intersection(
        derive_material_tags(item)
    ):
        return False
    return True


def apply_hard_filters(
    candidates: Iterable[Candidate],
    constraints: FilterConstraints,
    *,
    excluded_parent_asins: Iterable[str] = (),
) -> list[Candidate]:
    """Apply explicit constraints without changing candidate order."""

    # Reserved for session-aware filtering in a later stage.
    del excluded_parent_asins
    normalized_constraints = _normalize_constraints(constraints)
    return [
        candidate
        for candidate in candidates
        if _matches_constraints(candidate, normalized_constraints)
    ]
