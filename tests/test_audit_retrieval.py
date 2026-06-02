from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.tools.audit_retrieval import (
    AuditSession,
    LazySettingsQueryTranslator,
    build_html_report,
    interactive_loop,
    load_channels,
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
    assert json_path.name.endswith("_001_query.json")
    assert html_path.name.endswith("_001_query.html")
    assert json.loads(json_path.read_text(encoding="utf-8")) == report
    assert "&lt;Guitar Tuner&gt;" in html_path.read_text(encoding="utf-8")


def test_user_report_includes_training_history(tmp_path: Path) -> None:
    session, _, user_channel = build_session(tmp_path)

    report, _, html_path = session.run("user", "U1")

    assert user_channel.calls == [("U1", 3)]
    assert report["user_history"]["popularity"] == [
        {"parent_asin": "SEEN", "user_id": "U1"}
    ]
    assert "用户训练历史" in html_path.read_text(encoding="utf-8")


def test_interactive_loop_accepts_query_and_user_and_writes_each_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    session, query_channel, user_channel = build_session(tmp_path)
    inputs = iter(["bad", "query guitar tuner", "user U1", ":quit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

    interactive_loop(session)

    assert query_channel.calls == [("guitar tuner", 3)]
    assert user_channel.calls == [("U1", 3)]
    assert len(list(tmp_path.rglob("*.json"))) == 2
    assert len(list(tmp_path.rglob("*.html"))) == 2
    assert "Invalid command" in capsys.readouterr().out


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


def test_load_channels_rejects_future_placeholders_before_loading_data() -> None:
    with pytest.raises(ValueError, match="bm25: BM25 retrieval is not implemented yet"):
        load_channels(
            scope="dev",
            channel_names=["bm25"],
            qdrant_url="http://127.0.0.1:6333",
            device="cpu",
        )


def test_build_html_report_renders_dense_document_in_folded_details() -> None:
    html = build_html_report(
        {
            "input_type": "query",
            "input": "guitar tuner",
            "scope": "dev",
            "top_k": 1,
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
