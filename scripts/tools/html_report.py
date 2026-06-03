"""Shared HTML rendering helpers for developer inspection reports."""

from __future__ import annotations

import html
from collections.abc import Mapping, Sequence
from typing import Any


SHARED_STYLES = """
    :root { color-scheme: light; font-family: Arial, "Microsoft YaHei", sans-serif; }
    body { margin: 0; background: #f5f7fa; color: #1f2937; }
    main { max-width: 1500px; margin: auto; padding: 24px; }
    h1 { margin: 0 0 8px; }
    .subtitle { color: #64748b; margin-bottom: 20px; }
    .query-banner { margin: 0 0 18px; padding: 18px; color: white; background: #1e3a8a; border-radius: 10px; }
    .query-text { margin: 8px 0; font-size: 24px; font-weight: bold; line-height: 1.35; }
    .translated-query { margin-top: 8px; color: #dbeafe; }
    .score-toolbar { display: flex; align-items: center; gap: 16px; margin: 0 0 10px; }
    button { padding: 9px 14px; color: white; background: #047857; border: 0; border-radius: 7px; cursor: pointer; }
    button:hover { background: #065f46; }
    .toolbar { position: sticky; top: 0; z-index: 5; background: #f5f7fa; padding: 10px 0; }
    input { box-sizing: border-box; width: 100%; padding: 12px; border: 1px solid #cbd5e1; border-radius: 8px; }
    select { box-sizing: border-box; width: 100%; min-width: 170px; padding: 8px; border: 1px solid #cbd5e1; border-radius: 6px; background: white; }
    .score-notes { min-width: 180px; padding: 8px; }
    .score-label { display: none; }
    .score-panel { display: grid; grid-template-columns: minmax(220px, 360px) minmax(260px, 1fr); gap: 10px; margin: 14px 0; padding: 12px; background: #eff6ff; border-radius: 8px; }
    .table-wrap { overflow: auto; max-height: 520px; background: white; border: 1px solid #dbe2ea; border-radius: 8px; }
    table { border-collapse: collapse; width: 100%; }
    th, td { padding: 8px 10px; border-bottom: 1px solid #e5e7eb; text-align: left; vertical-align: top; }
    th { background: #f8fafc; white-space: nowrap; }
    .overview th { position: sticky; top: 0; }
    .item-card { margin: 24px 0; padding: 18px; background: white; border: 1px solid #dbe2ea; border-radius: 10px; }
    .item-card header { display: flex; justify-content: space-between; gap: 12px; }
    .number { display: inline-block; color: white; background: #1d4ed8; padding: 3px 7px; border-radius: 999px; }
    .fields th { width: 180px; }
    .nested th { width: 240px; }
    ul { margin: 0; padding-left: 20px; }
    details { margin-top: 14px; }
    summary { cursor: pointer; color: #1d4ed8; font-weight: bold; }
    pre { white-space: pre-wrap; word-break: break-word; background: #f8fafc; padding: 10px; }
    code { word-break: break-all; }
    .missing { color: #b91c1c; font-style: italic; }
    .hidden { display: none; }
"""


SEARCH_SCRIPT = """
  const search = document.getElementById("search");
  const cards = [...document.querySelectorAll(".item-card")];
  search.addEventListener("input", () => {
    const query = search.value.trim().toLocaleLowerCase();
    cards.forEach(card => card.classList.toggle("hidden", query && !card.dataset.search.includes(query)));
  });
"""


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
