"""Audit retrieval channels with reusable models and HTML plus JSON reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from cartwise.core.config import Settings
from cartwise.core.llm import QueryTranslationError, create_query_translator
from cartwise.retrieval.bm25 import BM25Index, BM25Retriever
from cartwise.retrieval.dense import (
    DENSE_MODEL_SPECS,
    DenseRetriever,
    collection_name,
    create_qdrant_client,
    load_dense_encoder,
)
from cartwise.retrieval.lightgcn import LightGCNRecommender
from cartwise.retrieval.popularity import PopularityRecommender
from scripts.paths import (
    ARTIFACT_REPORTS_ROOT,
    MODELS_ROOT,
    PRODUCT_BM25_ARTIFACT_ROOT,
    PROCESSED_ROOTS,
    RETRIEVAL_AUDIT_ARTIFACT_ROOT,
)
from scripts.tools.html_report import SEARCH_SCRIPT, SHARED_STYLES, escape, render_value
from scripts.tools.item_metadata import load_items_by_parent_asin


EXIT_COMMANDS = {":exit", ":quit", "exit", "quit"}
QUERY_CATALOG_PATH = ARTIFACT_REPORTS_ROOT / "manual_testing" / "retrieval_audit_queries.json"
QUERY_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
QUERY_CHANNELS = ("e5", "blair", "bm25")
USER_CHANNELS = ("popularity", "lightgcn")
UNAVAILABLE_CHANNELS = {
    "fusion": "stage-seven fusion is not implemented yet",
}
CHANNELS = (*USER_CHANNELS, *QUERY_CHANNELS, *UNAVAILABLE_CHANNELS)


def load_query_catalog(path: Path = QUERY_CATALOG_PATH) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("retrieval audit query catalog must be a JSON object")
    catalog: dict[str, str] = {}
    for query_id, query in payload.items():
        if not isinstance(query_id, str) or QUERY_ID_PATTERN.fullmatch(query_id) is None:
            raise ValueError(f"invalid retrieval audit query ID: {query_id}")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"retrieval audit query must not be empty: {query_id}")
        catalog[query_id.upper()] = query.strip()
    return catalog


def resolve_catalog_query(query_id: str, catalog: Mapping[str, str]) -> tuple[str, str]:
    normalized = query_id.strip().upper()
    if QUERY_ID_PATTERN.fullmatch(normalized) is None:
        raise ValueError("query ID must use only letters, numbers, hyphens, and underscores")
    try:
        return normalized, catalog[normalized]
    except KeyError as error:
        raise ValueError(
            f"unknown retrieval audit query ID: {normalized}; use 'query-id <ID>'"
        ) from error


def free_query_id(query: str) -> str:
    digest = hashlib.sha256(query.strip().encode("utf-8")).hexdigest()[:10].upper()
    return f"FREE-{digest}"


class AuditChannel(Protocol):
    input_type: str

    def recall(self, value: str, *, k: int) -> list[dict[str, Any]]: ...

    def history(self, value: str) -> list[dict[str, Any]]: ...


class LazySettingsQueryTranslator:
    """Delay external LLM client creation until the first Chinese query."""

    def __init__(self) -> None:
        self._translator = None

    def translate(self, query: str) -> str:
        if self._translator is None:
            self._translator = create_query_translator(Settings())
        return self._translator.translate(query)


def _item(items_by_parent_asin: Mapping[str, dict[str, Any]], parent_asin: str) -> dict[str, Any]:
    return dict(items_by_parent_asin.get(parent_asin, {"parent_asin": parent_asin}))


class PopularityAuditChannel:
    input_type = "user"

    def __init__(
        self,
        recommender: PopularityRecommender,
        items_by_parent_asin: Mapping[str, dict[str, Any]],
    ) -> None:
        self.recommender = recommender
        self.items_by_parent_asin = items_by_parent_asin

    def recall(self, value: str, *, k: int) -> list[dict[str, Any]]:
        return [
            {
                "channel": "popularity",
                "rank": rank,
                "parent_asin": parent_asin,
                "score": self.recommender.item_counts[parent_asin],
                "score_type": "interaction_count",
                "item": _item(self.items_by_parent_asin, parent_asin),
                "retrieval_query": None,
            }
            for rank, parent_asin in enumerate(
                self.recommender.recommend(value, k=k),
                start=1,
            )
        ]

    def history(self, value: str) -> list[dict[str, Any]]:
        return [
            _item(self.items_by_parent_asin, parent_asin)
            for parent_asin in sorted(
                self.recommender.interacted_items_by_user.get(value, ())
            )
        ]


class LightGCNAuditChannel:
    input_type = "user"

    def __init__(
        self,
        recommender: LightGCNRecommender,
        items_by_parent_asin: Mapping[str, dict[str, Any]],
    ) -> None:
        self.recommender = recommender
        self.items_by_parent_asin = items_by_parent_asin

    def recall(self, value: str, *, k: int) -> list[dict[str, Any]]:
        return [
            {
                "channel": "lightgcn",
                "rank": rank,
                "parent_asin": parent_asin,
                "score": None,
                "score_type": None,
                "item": _item(self.items_by_parent_asin, parent_asin),
                "retrieval_query": None,
            }
            for rank, parent_asin in enumerate(
                self.recommender.recommend(value, k=k),
                start=1,
            )
        ]

    def history(self, value: str) -> list[dict[str, Any]]:
        return [
            _item(self.items_by_parent_asin, parent_asin)
            for parent_asin in sorted(
                self.recommender.interacted_items_by_user.get(value, ())
            )
        ]


class DenseAuditChannel:
    input_type = "query"

    def __init__(
        self,
        channel: str,
        retriever: DenseRetriever,
        items_by_parent_asin: Mapping[str, dict[str, Any]],
    ) -> None:
        self.channel = channel
        self.retriever = retriever
        self.items_by_parent_asin = items_by_parent_asin

    def recall(self, value: str, *, k: int) -> list[dict[str, Any]]:
        return [
            {
                "channel": self.channel,
                "rank": rank,
                "parent_asin": result["parent_asin"],
                "score": result["dense_score"],
                "score_type": "dense_score",
                "item": _item(self.items_by_parent_asin, result["parent_asin"]),
                "retrieval_query": result["retrieval_query"],
                "document": result.get("document"),
            }
            for rank, result in enumerate(self.retriever.search(value, k=k), start=1)
        ]

    def history(self, value: str) -> list[dict[str, Any]]:
        del value
        return []


class BM25AuditChannel:
    input_type = "query"

    def __init__(
        self,
        retriever: BM25Retriever,
        items_by_parent_asin: Mapping[str, dict[str, Any]],
    ) -> None:
        self.retriever = retriever
        self.items_by_parent_asin = items_by_parent_asin

    def recall(self, value: str, *, k: int) -> list[dict[str, Any]]:
        return [
            {
                "channel": "bm25",
                "rank": rank,
                "parent_asin": result["parent_asin"],
                "score": result["bm25_score"],
                "score_type": "bm25_score",
                "item": _item(self.items_by_parent_asin, result["parent_asin"]),
                "retrieval_query": result["retrieval_query"],
                "document": result["document"],
            }
            for rank, result in enumerate(self.retriever.search(value, k=k), start=1)
        ]

    def history(self, value: str) -> list[dict[str, Any]]:
        del value
        return []


def load_channels(
    *,
    scope: str,
    channel_names: list[str],
    qdrant_url: str,
    device: str,
) -> dict[str, AuditChannel]:
    unavailable = [name for name in channel_names if name in UNAVAILABLE_CHANNELS]
    if unavailable:
        messages = ", ".join(f"{name}: {UNAVAILABLE_CHANNELS[name]}" for name in unavailable)
        raise ValueError(f"unavailable retrieval channel(s): {messages}")

    processed_root = PROCESSED_ROOTS[scope]
    items_by_parent_asin = load_items_by_parent_asin(processed_root / "items.parquet")
    channels: dict[str, AuditChannel] = {}
    training_path = processed_root / "interactions_train.parquet"
    if "popularity" in channel_names:
        channels["popularity"] = PopularityAuditChannel(
            PopularityRecommender.from_parquet(training_path),
            items_by_parent_asin,
        )
    if "lightgcn" in channel_names:
        channels["lightgcn"] = LightGCNAuditChannel(
            LightGCNRecommender.load(
                MODELS_ROOT / "lightgcn" / scope / "lightgcn.pt",
                device=device,
            ),
            items_by_parent_asin,
        )

    translator = LazySettingsQueryTranslator()
    if "bm25" in channel_names:
        index_path = PRODUCT_BM25_ARTIFACT_ROOT / scope / "bm25.json.gz"
        print(f"Loading bm25: {index_path}")
        channels["bm25"] = BM25AuditChannel(
            BM25Retriever(BM25Index.load(index_path), translator=translator),
            items_by_parent_asin,
        )

    dense_names = [name for name in channel_names if name in DENSE_MODEL_SPECS]
    if dense_names:
        client = create_qdrant_client(qdrant_url)
        for model_key in dense_names:
            collection = collection_name(scope, model_key)
            info = client.get_collection(collection)
            print(
                f"Loading {model_key}: {DENSE_MODEL_SPECS[model_key].model_name} "
                f"({info.points_count:,} indexed products)"
            )
            channels[model_key] = DenseAuditChannel(
                model_key,
                DenseRetriever(
                    client,
                    collection=collection,
                    encoder=load_dense_encoder(model_key, device=device),
                    translator=translator,
                ),
                items_by_parent_asin,
            )
    return {name: channels[name] for name in channel_names}


def _score_text(result: Mapping[str, Any]) -> str:
    score = result["score"]
    if score is None:
        return "not exposed"
    if isinstance(score, float):
        return f"{score:.6f}"
    return str(score)


def _score_record_id(channel: str, result: Mapping[str, Any]) -> str:
    return f"{channel}:{result['rank']}:{result['parent_asin']}"


def _relevance_control(record_id: str, *, location: str) -> str:
    control_id = f"relevance-{location}-{record_id}"
    return f"""
      <label class="score-label" for="{escape(control_id)}">相关度</label>
      <select id="{escape(control_id)}" class="score-control" data-record-id="{escape(record_id)}" data-field="human_relevance">
        <option value="">未评分</option>
        <option value="2">2 - 直接满足需求</option>
        <option value="1">1 - 部分相关或可接受替代</option>
        <option value="0">0 - 无关或明显错误</option>
      </select>
    """


def _notes_control(record_id: str, *, location: str) -> str:
    control_id = f"notes-{location}-{record_id}"
    return f"""
      <label class="score-label" for="{escape(control_id)}">备注</label>
      <input id="{escape(control_id)}" class="score-control score-notes" data-record-id="{escape(record_id)}" data-field="notes" placeholder="可选备注">
    """


def _script_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


def _build_scoring_script(report: Mapping[str, Any], score_rows: Sequence[Mapping[str, Any]]) -> str:
    return f"""
  const scoreRows = {_script_json(score_rows)};
  const reportId = {_script_json(report["report_id"])};
  const storageKey = `cartwise-retrieval-audit:${{reportId}}`;
  const scoreControls = [...document.querySelectorAll(".score-control")];
  let savedScores = {{}};

  try {{
    savedScores = JSON.parse(localStorage.getItem(storageKey) || "{{}}");
  }} catch (error) {{
    savedScores = {{}};
  }}

  function saveScores() {{
    try {{
      localStorage.setItem(storageKey, JSON.stringify(savedScores));
    }} catch (error) {{
      console.warn("Unable to persist retrieval audit scores", error);
    }}
  }}

  function updateProgress() {{
    const scored = scoreRows.filter(row => savedScores[row.record_id]?.human_relevance !== undefined && savedScores[row.record_id]?.human_relevance !== "").length;
    document.getElementById("score-progress").textContent = `已评分 ${{scored}} / ${{scoreRows.length}}`;
  }}

  function synchronizeControls(recordId, field, value) {{
    scoreControls
      .filter(control => control.dataset.recordId === recordId && control.dataset.field === field)
      .forEach(control => {{ control.value = value; }});
  }}

  scoreControls.forEach(control => {{
    const record = savedScores[control.dataset.recordId] || {{}};
    control.value = record[control.dataset.field] || "";
    control.addEventListener("input", () => {{
      const recordId = control.dataset.recordId;
      const field = control.dataset.field;
      savedScores[recordId] = savedScores[recordId] || {{}};
      savedScores[recordId][field] = control.value;
      synchronizeControls(recordId, field, control.value);
      saveScores();
      updateProgress();
    }});
  }});

  function csvCell(value) {{
    const text = value === null || value === undefined ? "" : String(value);
    return `"${{text.replaceAll('"', '""')}}"`;
  }}

  document.getElementById("export-scores").addEventListener("click", () => {{
    const headers = [
      "report_id", "generated_at", "scope", "query_id", "input", "retrieval_query",
      "channel", "rank", "parent_asin", "title", "brand", "price",
      "retrieval_score", "score_type", "human_relevance", "notes"
    ];
    const csvRows = scoreRows.map(row => {{
      const score = savedScores[row.record_id] || {{}};
      const output = {{
        report_id: reportId,
        generated_at: {_script_json(report["generated_at"])},
        scope: {_script_json(report["scope"])},
        query_id: {_script_json(report["query_id"])},
        input: {_script_json(report["input"])},
        ...row,
        human_relevance: score.human_relevance || "",
        notes: score.notes || ""
      }};
      return headers.map(header => csvCell(output[header])).join(",");
    }});
    const csv = "\\ufeff" + [headers.map(csvCell).join(","), ...csvRows].join("\\r\\n") + "\\r\\n";
    const link = document.createElement("a");
    link.href = URL.createObjectURL(new Blob([csv], {{ type: "text/csv;charset=utf-8" }}));
    link.download = `${{reportId}}_scores.csv`;
    link.click();
    URL.revokeObjectURL(link.href);
  }});

  updateProgress();
"""


def _retrieval_query_html(report: Mapping[str, Any]) -> str:
    retrieval_queries = {
        result["retrieval_query"]
        for results in report["results"].values()
        for result in results
        if result.get("retrieval_query") and result["retrieval_query"] != report["input"]
    }
    if not retrieval_queries:
        return ""
    return (
        '<div class="translated-query"><strong>实际检索文本：</strong>'
        f"{escape(' | '.join(sorted(retrieval_queries)))}</div>"
    )


def build_html_report(report: Mapping[str, Any]) -> str:
    rows: list[str] = []
    cards: list[str] = []
    score_rows: list[dict[str, Any]] = []
    scoring_enabled = report["input_type"] == "query"
    card_index = 0
    for channel, results in report["results"].items():
        for result in results:
            card_index += 1
            item = result["item"]
            price = item.get("price")
            price_text = "missing" if price is None else f"${price:,.2f}"
            record_id = _score_record_id(channel, result)
            scoring_cells = ""
            scoring_panel = ""
            if scoring_enabled:
                score_rows.append(
                    {
                        "record_id": record_id,
                        "retrieval_query": result["retrieval_query"],
                        "channel": channel,
                        "rank": result["rank"],
                        "parent_asin": result["parent_asin"],
                        "title": item.get("title"),
                        "brand": item.get("brand"),
                        "price": price,
                        "retrieval_score": result["score"],
                        "score_type": result["score_type"],
                    }
                )
                scoring_cells = (
                    f"<td>{_relevance_control(record_id, location=f'table-{card_index}')}</td>"
                    f"<td>{_notes_control(record_id, location=f'table-{card_index}')}</td>"
                )
                scoring_panel = (
                    '<div class="score-panel">'
                    f"{_relevance_control(record_id, location=f'card-{card_index}')}"
                    f"{_notes_control(record_id, location=f'card-{card_index}')}"
                    "</div>"
                )
            rows.append(
                f'<tr><td><a href="#item-{card_index}">{escape(result["rank"])}</a></td>'
                f"<td>{escape(channel)}</td>"
                f"<td>{escape(_score_text(result))}</td>"
                f"<td>{escape(result['parent_asin'])}</td>"
                f"<td>{escape(item.get('title') or 'missing')}</td>"
                f"<td>{escape(item.get('brand') or 'missing')}</td>"
                f"<td>{escape(price_text)}</td>"
                f"{scoring_cells}</tr>"
            )
            details = {
                "score_type": result["score_type"],
                "retrieval_query": result["retrieval_query"],
                "item": item,
            }
            if "document" in result:
                details["document"] = result["document"]
            cards.append(
                f"""
    <article class="item-card" id="item-{card_index}" data-search="{escape(json.dumps(result, ensure_ascii=False).casefold())}">
      <header>
        <div><span class="number">#{escape(result["rank"])}</span> <strong>{escape(channel)}</strong> <code>{escape(result["parent_asin"])}</code></div>
        <a href="#top">back to top</a>
      </header>
      <h2>{escape(item.get("title") or "missing title")}</h2>
      {scoring_panel}
      <details><summary>完整元数据与检索信息</summary>{render_value(details)}</details>
    </article>
                """
            )
    history = report.get("user_history")
    history_html = ""
    if history is not None:
        history_html = (
            "<details><summary>用户训练历史</summary>"
            f"{render_value(history)}</details>"
        )
    query_banner_html = ""
    scoring_toolbar_html = ""
    scoring_script = ""
    scoring_headers = ""
    if scoring_enabled:
        channel_text = "_".join(report["channels"])
        query_banner_html = f"""
  <section class="query-banner">
    <div><strong>查询 ID：</strong>{escape(report["query_id"])}</div>
    <div class="query-text">{escape(report["input"])}</div>
    <div><strong>模型：</strong>{escape(channel_text)}</div>
    {_retrieval_query_html(report)}
  </section>
"""
        scoring_toolbar_html = """
  <div class="score-toolbar">
    <strong id="score-progress">已评分 0 / 0</strong>
    <button id="export-scores" type="button">导出评分 CSV</button>
  </div>
"""
        scoring_script = _build_scoring_script(report, score_rows)
        scoring_headers = "<th>相关度</th><th>备注</th>"
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CartWise 召回审核报告</title>
  <style>{SHARED_STYLES}</style>
</head>
<body>
<main id="top">
  <h1>CartWise 召回审核报告</h1>
  <div class="subtitle">输入类型：{escape(report["input_type"])} · 输入：{escape(report["input"])} · scope：{escape(report["scope"])} · Top K：{escape(report["top_k"])}</div>
  {query_banner_html}
  {history_html}
  {scoring_toolbar_html}
  <div class="toolbar"><input id="search" placeholder="搜索通道、ASIN、标题、品牌或任意属性值"></div>
  <div class="table-wrap">
    <table class="overview">
      <thead><tr><th>rank</th><th>channel</th><th>score</th><th>parent_asin</th><th>title</th><th>brand</th><th>price</th>{scoring_headers}</tr></thead>
      <tbody>{"".join(rows)}</tbody>
    </table>
  </div>
  <section id="cards">{"".join(cards)}</section>
</main>
<script>{SEARCH_SCRIPT}</script>
<script>{scoring_script}</script>
</body>
</html>
"""


class AuditSession:
    def __init__(
        self,
        channels: Mapping[str, AuditChannel],
        *,
        scope: str,
        top_k: int,
        output_root: Path,
        output_prefix: Path | None = None,
    ) -> None:
        if top_k <= 0:
            raise ValueError("--top-k must be greater than zero")
        self.channels = dict(channels)
        self.scope = scope
        self.top_k = top_k
        self.output_root = output_root
        self.output_prefix = output_prefix
        self.sequence = 0

    def _output_stem(
        self,
        input_type: str,
        *,
        channel_names: Sequence[str],
        query_id: str | None,
    ) -> Path:
        self.sequence += 1
        if self.output_prefix is not None:
            if self.sequence == 1:
                return self.output_prefix
            return self.output_prefix.with_name(
                f"{self.output_prefix.name}_{self.sequence:03d}_{input_type}"
            )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        channels = "_".join(channel_names)
        identifier = query_id if input_type == "query" else input_type
        return (
            self.output_root
            / self.scope
            / f"{timestamp}_{self.sequence:03d}_{channels}_{identifier}"
        )

    def run(
        self,
        input_type: str,
        value: str,
        *,
        query_id: str | None = None,
    ) -> tuple[dict[str, Any], Path, Path]:
        normalized = value.strip()
        if input_type not in {"query", "user"}:
            raise ValueError(f"unsupported input type: {input_type}")
        if not normalized:
            raise ValueError(f"{input_type} must not be empty")
        if input_type == "query":
            normalized_query_id = query_id.strip().upper() if query_id else free_query_id(normalized)
            if QUERY_ID_PATTERN.fullmatch(normalized_query_id) is None:
                raise ValueError(
                    "query ID must use only letters, numbers, hyphens, and underscores"
                )
        else:
            if query_id is not None:
                raise ValueError("query ID can only be used with query input")
            normalized_query_id = None
        selected = {
            name: channel
            for name, channel in self.channels.items()
            if channel.input_type == input_type
        }
        if not selected:
            raise ValueError(f"no selected retrieval channels accept {input_type} input")
        report: dict[str, Any] = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "scope": self.scope,
            "top_k": self.top_k,
            "input_type": input_type,
            "input": normalized,
            "query_id": normalized_query_id,
            "channels": list(selected),
            "results": {
                name: channel.recall(normalized, k=self.top_k)
                for name, channel in selected.items()
            },
        }
        if input_type == "user":
            report["user_history"] = {
                name: channel.history(normalized) for name, channel in selected.items()
            }
        stem = self._output_stem(
            input_type,
            channel_names=list(selected),
            query_id=normalized_query_id,
        )
        report["report_id"] = stem.name
        json_path = stem.with_suffix(".json")
        html_path = stem.with_suffix(".html")
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        html_path.write_text(build_html_report(report), encoding="utf-8")
        return report, json_path, html_path


def print_report_paths(json_path: Path, html_path: Path) -> None:
    print(f"Wrote JSON: {json_path}")
    print(f"Wrote HTML: {html_path}")


def interactive_loop(
    session: AuditSession,
    *,
    query_catalog: Mapping[str, str] | None = None,
) -> None:
    print("Channels are ready. Enter 'query-id <ID>', 'query <text>', 'user <id>', or :quit.")
    catalog = query_catalog or load_query_catalog()
    while True:
        try:
            command = input("\naudit> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if command.casefold() in EXIT_COMMANDS:
            return
        if not command:
            continue
        input_type, separator, value = command.partition(" ")
        if input_type not in {"query-id", "query", "user"} or not separator or not value.strip():
            print("Invalid command. Use 'query-id <ID>', 'query <text>', 'user <id>', or :quit.")
            continue
        try:
            if input_type == "query-id":
                query_id, query = resolve_catalog_query(value, catalog)
                _, json_path, html_path = session.run("query", query, query_id=query_id)
            else:
                _, json_path, html_path = session.run(input_type, value)
        except (QueryTranslationError, ValueError) as error:
            print(f"Audit error: {error}")
            continue
        print_report_paths(json_path, html_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=PROCESSED_ROOTS, default="full")
    parser.add_argument("--channels", nargs="+", choices=CHANNELS, required=True)
    query_group = parser.add_mutually_exclusive_group()
    query_group.add_argument("--query")
    query_group.add_argument("--query-id")
    parser.add_argument("--user-id")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--qdrant-url")
    parser.add_argument("--output-prefix", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings()
    try:
        channels = load_channels(
            scope=args.scope,
            channel_names=args.channels,
            qdrant_url=args.qdrant_url or settings.qdrant_url,
            device=args.device,
        )
        session = AuditSession(
            channels,
            scope=args.scope,
            top_k=args.top_k,
            output_root=RETRIEVAL_AUDIT_ARTIFACT_ROOT,
            output_prefix=args.output_prefix,
        )
        if args.query is None and args.query_id is None and args.user_id is None:
            interactive_loop(session)
            return
        if args.query_id is not None:
            query_id, query = resolve_catalog_query(args.query_id, load_query_catalog())
            _, json_path, html_path = session.run("query", query, query_id=query_id)
            print_report_paths(json_path, html_path)
        if args.query is not None:
            _, json_path, html_path = session.run("query", args.query)
            print_report_paths(json_path, html_path)
        if args.user_id is not None:
            _, json_path, html_path = session.run("user", args.user_id)
            print_report_paths(json_path, html_path)
    except (QueryTranslationError, ValueError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
