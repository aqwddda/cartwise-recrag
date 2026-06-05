"""Shared query intent and filter constraint types."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass


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
