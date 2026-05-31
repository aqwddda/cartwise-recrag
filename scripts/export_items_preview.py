"""Export a readable HTML preview of processed product metadata."""

from __future__ import annotations

import argparse
import html
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "items.parquet"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "generated" / "items_preview_first_100.html"
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


def escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def render_value(value: Any) -> str:
    if value is None or value == "" or value == []:
        return '<span class="missing">missing</span>'
    if isinstance(value, Mapping):
        rows = "".join(
            f"<tr><th>{escape(key)}</th><td>{render_value(item)}</td></tr>"
            for key, item in sorted(value.items())
        )
        return f'<table class="nested"><tbody>{rows}</tbody></table>'
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        items = "".join(f"<li>{render_value(item)}</li>" for item in value)
        return f"<ul>{items}</ul>"
    return f"<span>{escape(value)}</span>"


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
    :root {{ color-scheme: light; font-family: Arial, "Microsoft YaHei", sans-serif; }}
    body {{ margin: 0; background: #f5f7fa; color: #1f2937; }}
    main {{ max-width: 1500px; margin: auto; padding: 24px; }}
    h1 {{ margin: 0 0 8px; }}
    .subtitle {{ color: #64748b; margin-bottom: 20px; }}
    .toolbar {{ position: sticky; top: 0; z-index: 5; background: #f5f7fa; padding: 10px 0; }}
    input {{ box-sizing: border-box; width: 100%; padding: 12px; border: 1px solid #cbd5e1; border-radius: 8px; }}
    .table-wrap {{ overflow: auto; max-height: 520px; background: white; border: 1px solid #dbe2ea; border-radius: 8px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid #e5e7eb; text-align: left; vertical-align: top; }}
    th {{ background: #f8fafc; white-space: nowrap; }}
    .overview th {{ position: sticky; top: 0; }}
    .item-card {{ margin: 24px 0; padding: 18px; background: white; border: 1px solid #dbe2ea; border-radius: 10px; }}
    .item-card header {{ display: flex; justify-content: space-between; gap: 12px; }}
    .number {{ display: inline-block; color: white; background: #1d4ed8; padding: 3px 7px; border-radius: 999px; }}
    .fields th {{ width: 180px; }}
    .nested th {{ width: 240px; }}
    ul {{ margin: 0; padding-left: 20px; }}
    details {{ margin-top: 14px; }}
    summary {{ cursor: pointer; color: #1d4ed8; font-weight: bold; }}
    pre {{ white-space: pre-wrap; word-break: break-word; background: #f8fafc; padding: 10px; }}
    code {{ word-break: break-all; }}
    .missing {{ color: #b91c1c; font-style: italic; }}
    .hidden {{ display: none; }}
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
  const search = document.getElementById("search");
  const cards = [...document.querySelectorAll(".item-card")];
  search.addEventListener("input", () => {{
    const query = search.value.trim().toLocaleLowerCase();
    cards.forEach(card => card.classList.toggle("hidden", query && !card.dataset.search.includes(query)));
  }});
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
