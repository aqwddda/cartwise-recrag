from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.tools.audit_retrieval import (
    AuditSession,
    LazySettingsQueryTranslator,
    build_html_report,
    free_query_id,
    interactive_loop,
    load_query_catalog,
    load_channels,
    parse_args,
    resolve_catalog_query,
)


class FakeChannel:
    def __init__(self, input_type: str) -> None:
        self.input_type = input_type
        self.calls: list[tuple[str, int]] = []

    def recall(self, value: str, *, k: int):
        self.calls.append((value, k))
        return [
            {
                "channel": self.input_type,
                "rank": 1,
                "parent_asin": "P1",
                "score": 0.9 if self.input_type == "query" else None,
                "score_type": "dense_score" if self.input_type == "query" else None,
                "item": {
                    "parent_asin": "P1",
                    "title": "<Guitar Tuner>",
                    "brand": "Example",
                    "price": 19.99,
                },
                "retrieval_query": value if self.input_type == "query" else None,
            }
        ]

    def history(self, value: str):
        return [{"parent_asin": "SEEN", "user_id": value}]


class FakeFusionChannel(FakeChannel):
    accepts_user_context = True

    def __init__(self) -> None:
        super().__init__("query")
        self.user_calls: list[tuple[str, int, str | None]] = []
        self.sidecar_written = False

    def recall(self, value: str, *, k: int, user_id: str | None = None):
        self.user_calls.append((value, k, user_id))
        return [
            {
                "channel": "fusion",
                "rank": 1,
                "parent_asin": "P1",
                "score": 0.02,
                "score_type": "weighted_rrf",
                "fusion_score": 0.02,
                "sources": ["dense", "bm25"],
                "source_ranks": {"dense": 1, "bm25": 2},
                "source_scores": {"dense": 0.9, "bm25": 12.0},
                "item": {
                    "parent_asin": "P1",
                    "title": "Fusion Tuner",
                    "brand": "Example",
                    "price": 19.99,
                },
                "retrieval_query": value,
            }
        ]

    def write_sidecar_reports(self, stem: Path):
        self.sidecar_written = True
        ranked = stem.with_name(f"{stem.name}_fusion_ranked.json")
        filtered = stem.with_name(f"{stem.name}_filtered.json")
        payload = {
            "fusion_intent": self.audit_metadata()["fusion_intent"],
            "filter_constraints": self.audit_metadata()["filter_constraints"],
            "results": [],
        }
        ranked.write_text(json.dumps(payload), encoding="utf-8")
        filtered.write_text(json.dumps(payload), encoding="utf-8")
        return {"fusion_ranked_json": ranked, "filtered_json": filtered}

    def audit_metadata(self):
        return {
            "fusion_intent": {
                "search_query": "guitar tuner",
                "product_terms": ["guitar tuner"],
                "raw_filters": {"max_price": 50.0},
            },
            "filter_constraints": {
                "category_tags": ["Accessories"],
                "min_price": None,
                "max_price": 50.0,
                "brands": [],
                "excluded_brands": ["Fender"],
                "color_tags": [],
                "material_tags": [],
            },
        }


def build_session(tmp_path: Path) -> tuple[AuditSession, FakeChannel, FakeChannel]:
    query_channel = FakeChannel("query")
    user_channel = FakeChannel("user")
    return (
        AuditSession(
            {"e5": query_channel, "popularity": user_channel},
            scope="dev",
            top_k=3,
            output_root=tmp_path,
        ),
        query_channel,
        user_channel,
    )


def test_session_runs_matching_channel_and_writes_json_and_html(tmp_path: Path) -> None:
    session, query_channel, user_channel = build_session(tmp_path)

    report, json_path, html_path = session.run("query", "guitar tuner")

    assert query_channel.calls == [("guitar tuner", 3)]
    assert user_channel.calls == []
    assert json_path.name.endswith(f"_001_e5_{free_query_id('guitar tuner')}.json")
    assert html_path.name.endswith(f"_001_e5_{free_query_id('guitar tuner')}.html")
    assert json.loads(json_path.read_text(encoding="utf-8")) == report
    assert "&lt;Guitar Tuner&gt;" in html_path.read_text(encoding="utf-8")
    assert report["report_id"] == html_path.stem
    assert report["query_id"] == free_query_id("guitar tuner")


def test_user_report_includes_training_history(tmp_path: Path) -> None:
    session, _, user_channel = build_session(tmp_path)

    report, _, html_path = session.run("user", "U1")

    assert user_channel.calls == [("U1", 3)]
    assert report["user_history"]["popularity"] == [
        {"parent_asin": "SEEN", "user_id": "U1"}
    ]
    html = html_path.read_text(encoding="utf-8")
    assert "用户训练历史" in html
    assert "导出评分 CSV" not in html


def test_interactive_loop_accepts_query_and_user_and_writes_each_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    session, query_channel, user_channel = build_session(tmp_path)
    inputs = iter(["bad", "query-id EN-01", "query guitar tuner", "user U1", ":quit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

    interactive_loop(session, query_catalog={"EN-01": "catalog tuner"})

    assert query_channel.calls == [("catalog tuner", 3), ("guitar tuner", 3)]
    assert user_channel.calls == [("U1", 3)]
    assert len(list(tmp_path.rglob("*.json"))) == 3
    assert len(list(tmp_path.rglob("*.html"))) == 3
    assert "Invalid command" in capsys.readouterr().out


def test_interactive_loop_accepts_user_query_for_fusion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    channel = FakeFusionChannel()
    session = AuditSession(
        {"fusion": channel},
        scope="dev",
        top_k=3,
        output_root=tmp_path,
    )
    inputs = iter(["user-query U1 guitar tuner under 50", ":quit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

    interactive_loop(session, query_catalog={})

    assert channel.user_calls == [("guitar tuner under 50", 3, "U1")]
    assert len(list(tmp_path.rglob("*.json"))) == 3
    assert len(list(tmp_path.rglob("*.html"))) == 1


def test_session_rejects_input_without_matching_channel(tmp_path: Path) -> None:
    session = AuditSession(
        {"e5": FakeChannel("query")},
        scope="dev",
        top_k=3,
        output_root=tmp_path,
    )

    with pytest.raises(ValueError, match="no selected retrieval channels"):
        session.run("user", "U1")


def test_custom_output_prefix_is_reused_for_later_reports(tmp_path: Path) -> None:
    channel = FakeChannel("query")
    session = AuditSession(
        {"e5": channel},
        scope="dev",
        top_k=3,
        output_root=tmp_path / "default",
        output_prefix=tmp_path / "custom" / "audit",
    )

    _, first_json, _ = session.run("query", "first")
    _, second_json, _ = session.run("query", "second")

    assert first_json == tmp_path / "custom" / "audit.json"
    assert second_json == tmp_path / "custom" / "audit_002_query.json"


def test_session_passes_user_context_to_fusion_and_writes_sidecars(tmp_path: Path) -> None:
    channel = FakeFusionChannel()
    session = AuditSession(
        {"fusion": channel},
        scope="dev",
        top_k=3,
        output_root=tmp_path,
    )

    report, json_path, html_path = session.run("query", "guitar tuner", user_id="U1")

    assert channel.user_calls == [("guitar tuner", 3, "U1")]
    assert channel.sidecar_written
    assert report["user_id"] == "U1"
    assert report["fusion_intent"]["product_terms"] == ["guitar tuner"]
    assert report["filter_constraints"]["category_tags"] == ["Accessories"]
    assert "fusion_fusion_ranked_json" in report["sidecar_reports"]
    assert "fusion_filtered_json" in report["sidecar_reports"]
    ranked_payload = json.loads(
        Path(report["sidecar_reports"]["fusion_fusion_ranked_json"]).read_text(
            encoding="utf-8"
        )
    )
    filtered_payload = json.loads(
        Path(report["sidecar_reports"]["fusion_filtered_json"]).read_text(
            encoding="utf-8"
        )
    )
    assert ranked_payload["filter_constraints"]["excluded_brands"] == ["Fender"]
    assert filtered_payload["fusion_intent"]["search_query"] == "guitar tuner"
    assert json.loads(json_path.read_text(encoding="utf-8")) == report
    html = html_path.read_text(encoding="utf-8")
    assert "Fusion Tuner" in html
    assert "source_ranks" in html
    assert "Fusion 意图与过滤约束" in html
    assert "filter_constraints" in html


def test_build_html_report_renders_dense_document_in_folded_details() -> None:
    html = build_html_report(
        {
            "generated_at": "2026-06-02T12:00:00",
            "report_id": "report",
            "query_id": "EN-01",
            "input_type": "query",
            "input": "guitar tuner",
            "scope": "dev",
            "top_k": 1,
            "channels": ["e5"],
            "results": {
                "e5": [
                    {
                        "channel": "e5",
                        "rank": 1,
                        "parent_asin": "P1",
                        "score": 0.9,
                        "score_type": "dense_score",
                        "item": {"parent_asin": "P1", "title": "Tuner", "price": None},
                        "retrieval_query": "guitar tuner",
                        "document": "Title: <Tuner>",
                    }
                ]
            },
        }
    )

    assert "完整元数据与检索信息" in html
    assert "Title: &lt;Tuner&gt;" in html
    assert "查询 ID：</strong>EN-01" in html
    assert "guitar tuner" in html
    assert "模型：</strong>e5" in html
    assert "导出评分 CSV" in html
    assert "2 - 直接满足需求" in html
    assert 'data-record-id="e5:1:P1"' in html
    assert "cartwise-retrieval-audit:${reportId}" in html


def test_html_report_shows_translated_query_when_it_differs_from_input() -> None:
    html = build_html_report(
        {
            "generated_at": "2026-06-02T12:00:00",
            "report_id": "report",
            "query_id": "ZH-01",
            "input_type": "query",
            "input": "吉他调音器",
            "scope": "dev",
            "top_k": 1,
            "channels": ["e5"],
            "results": {
                "e5": [
                    {
                        "channel": "e5",
                        "rank": 1,
                        "parent_asin": "P1",
                        "score": 0.9,
                        "score_type": "dense_score",
                        "item": {"parent_asin": "P1", "title": "Tuner", "price": None},
                        "retrieval_query": "guitar tuner",
                    }
                ]
            },
        }
    )

    assert "实际检索文本：</strong>guitar tuner" in html


def test_query_catalog_contains_expected_manual_evaluation_queries() -> None:
    catalog = load_query_catalog()

    assert len(catalog) == 30
    assert catalog["EN-01"] == "guitar tuner for beginners"
    assert catalog["ZH-05"] == "我想给刚开始学吉他的朋友买一个实用的小礼物"
    assert resolve_catalog_query("en-01", catalog) == (
        "EN-01",
        "guitar tuner for beginners",
    )


def test_unknown_query_catalog_id_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown retrieval audit query ID"):
        resolve_catalog_query("EN-99", {"EN-01": "guitar tuner"})


def test_free_query_id_is_stable_and_filename_safe() -> None:
    assert free_query_id(" guitar tuner ") == free_query_id("guitar tuner")
    assert free_query_id("guitar tuner").startswith("FREE-")


def test_default_output_filename_includes_all_query_channels(tmp_path: Path) -> None:
    session = AuditSession(
        {"e5": FakeChannel("query"), "blair": FakeChannel("query"), "bm25": FakeChannel("query")},
        scope="dev",
        top_k=3,
        output_root=tmp_path,
    )

    _, json_path, _ = session.run("query", "guitar tuner", query_id="EN-01")

    assert json_path.name.endswith("_001_e5_blair_bm25_EN-01.json")


def test_parse_args_defaults_to_ten_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.argv", ["audit_retrieval", "--channels", "e5"])

    args = parse_args()

    assert args.top_k == 10
    assert args.dense_k == 30
    assert args.bm25_k == 30
    assert args.lightgcn_k == 30
    assert args.popularity_k == 30
    assert args.rrf_k == 60


def test_lazy_translator_creates_external_client_only_when_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class FakeTranslator:
        def translate(self, query: str) -> str:
            calls.append(query)
            return "guitar tuner"

    monkeypatch.setattr(
        "scripts.tools.audit_retrieval.create_query_translator",
        lambda settings: FakeTranslator(),
    )
    translator = LazySettingsQueryTranslator()

    assert calls == []
    assert translator.translate("吉他调音器") == "guitar tuner"
    assert calls == ["吉他调音器"]
