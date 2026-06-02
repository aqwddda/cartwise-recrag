"""Shared product metadata loading for developer tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def load_items_by_parent_asin(
    path: Path,
    *,
    columns: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    return {
        item["parent_asin"]: item
        for item in pq.read_table(path, columns=columns).to_pylist()
    }
