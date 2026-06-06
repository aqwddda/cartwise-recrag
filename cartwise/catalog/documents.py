"""Shared product document rendering for lexical and dense retrieval."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any


def _render_sequence(value: Any) -> str:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ""
    return " | ".join(str(entry).strip() for entry in value if str(entry).strip())


def _render_details(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        return ""
    try:
        details = json.loads(value)
    except json.JSONDecodeError:
        return value.strip()
    if not isinstance(details, Mapping):
        return ""
    return " | ".join(
        f"{key}: {json.dumps(detail, ensure_ascii=False, sort_keys=True)}"
        for key, detail in sorted(details.items())
    )


def _render_scalar(value: Any) -> str:
    return "" if value is None else str(value).strip()


def build_product_document(item: Mapping[str, Any]) -> str:
    """Build the shared E5, BLaIR, and BM25 product document."""

    fields = [
        ("Title", _render_scalar(item.get("title"))),
        ("Brand", _render_scalar(item.get("brand"))),
        ("Main Category", _render_scalar(item.get("main_category"))),
        ("Categories", _render_sequence(item.get("categories"))),
        ("Features", _render_sequence(item.get("features"))),
        ("Details", _render_details(item.get("details_json"))),
        ("Description", _render_scalar(item.get("description"))),
    ]
    return "\n".join(f"{name}: {value}" for name, value in fields)
