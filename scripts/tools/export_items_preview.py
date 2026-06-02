"""Export a readable HTML preview of processed product metadata."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from scripts.paths import ARTIFACT_PREVIEWS_ROOT, PROCESSED_ROOT
from scripts.tools.html_report import SEARCH_SCRIPT, SHARED_STYLES, escape, render_value


DEFAULT_INPUT = PROCESSED_ROOT / "items.parquet"
DEFAULT_OUTPUT = ARTIFACT_PREVIEWS_ROOT / "items_preview_first_100.html"
TOP_LEVEL_FIELDS = (
    "parent_asin",
    "title",
    "brand",
    "price",
    "main_category",
    "categories",
    "description",
    "features",
    "bought_together",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=100)
    return parser.parse_args()


def render_overview_row(index: int, row: Mapping[str, Any]) -> str:
    details = json.loads(row["details_json"] or "{}")
    price = "missing" if row["price"] is None else f"${row['price']:,.2f}"
    return (
        f'<tr><td><a href="#item-{index}">{index}</a></td>'
        f"<td>{escape(row['parent_asin'])}</td>"
        f"<td>{escape(row['title'] or 'missing')}</td>"
        f"<td>{escape(row['brand'] or 'missing')}</td>"
        f"<td>{escape(price)}</td>"
        f"<td>{escape(row['main_category'] or 'missing')}</td>"
        f"<td>{len(details)}</td></tr>"
    )


def render_item(index: int, row: Mapping[str, Any]) -> str:
    fields = "".join(
        f"<tr><th>{escape(field)}</th><td>{render_value(row[field])}</td></tr>"
        for field in TOP_LEVEL_FIELDS
    )
    details = json.loads(row["details_json"] or "{}")
    return f"""
    <article class="item-card" id="item-{index}" data-search="{escape(json.dumps(row, ensure_ascii=False).casefold())}">
      <header>
        <div><span class="number">#{index}</span> <code>{escape(row["parent_asin"])}</code></div>
        <a href="#top">back to top</a>
      </header>
      <h2>{escape(row["title"] or "missing title")}</h2>
      <table class="fields"><tbody>{fields}</tbody></table>
      <details open>
        <summary>details_json expanded ({len(details)} attributes)</summary>
        {render_value(details)}
      </details>
      <details>
        <summary>details_json raw</summary>
        <pre>{escape(row["details_json"] or "{}")}</pre>
      </details>
    </article>
    """


def build_report(rows: list[dict[str, Any]], source: Path) -> str:
    overview = "".join(render_overview_row(index, row) for index, row in enumerate(rows, 1))
    cards = "".join(render_item(index, row) for index, row in enumerate(rows, 1))
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CartWise 商品元数据预览</title>
  <style>
{SHARED_STYLES}
  </style>
</head>
<body>
<main id="top">
  <h1>CartWise 商品元数据预览</h1>
  <div class="subtitle">来源：{escape(source)} · 展示前 {len(rows)} 条商品 · 顶层字段与 details_json 完整展开</div>
  <div class="toolbar"><input id="search" placeholder="搜索 ASIN、标题、品牌或任意属性值，例如 Fender、USB、Guitar"></div>
  <h2>总览对比</h2>
  <div class="table-wrap">
    <table class="overview">
      <thead><tr><th>#</th><th>parent_asin</th><th>title</th><th>brand</th><th>price</th><th>main_category</th><th>details 属性数</th></tr></thead>
      <tbody>{overview}</tbody>
    </table>
  </div>
  <h2>完整详情</h2>
  <section id="cards">{cards}</section>
</main>
<script>
{SEARCH_SCRIPT}
</script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    if args.limit <= 0:
        raise ValueError("--limit must be greater than zero")
    rows = pq.read_table(args.input).slice(0, args.limit).to_pylist()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_report(rows, args.input), encoding="utf-8")
    print(f"Wrote {len(rows)} items to {args.output}")


if __name__ == "__main__":
    main()
