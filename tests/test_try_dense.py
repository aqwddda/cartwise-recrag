from __future__ import annotations

from scripts.tools.compare_dense_models import (
    LazySettingsQueryTranslator,
    interactive_loop,
    search_all,
)


class FakeRetriever:
    def __init__(self, model_key: str) -> None:
        self.model_key = model_key
        self.queries: list[tuple[str, int]] = []

    def search(self, query: str, *, k: int):
        self.queries.append((query, k))
        return [
            {
                "parent_asin": "P1",
                "dense_score": 0.9,
                "title": f"{self.model_key} result",
                "retrieval_query": query,
            }
        ]


def test_search_all_reuses_loaded_retrievers() -> None:
    retrievers = {
        "e5": FakeRetriever("e5"),
        "blair": FakeRetriever("blair"),
    }

    results = search_all(retrievers, "guitar tuner", top_k=3)

    assert results["e5"][0]["title"] == "e5 result"
    assert results["blair"][0]["title"] == "blair result"
    assert retrievers["e5"].queries == [("guitar tuner", 3)]
    assert retrievers["blair"].queries == [("guitar tuner", 3)]


def test_interactive_loop_queries_until_quit(monkeypatch, capsys) -> None:
    retriever = FakeRetriever("blair")
    inputs = iter(["", "guitar tuner", ":quit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

    interactive_loop({"blair": retriever}, top_k=5)

    assert retriever.queries == [("guitar tuner", 5)]
    assert "[blair] guitar tuner" in capsys.readouterr().out


def test_lazy_translator_creates_external_client_only_when_used(monkeypatch) -> None:
    calls: list[str] = []

    class FakeTranslator:
        def translate(self, query: str) -> str:
            calls.append(query)
            return "guitar tuner"

    monkeypatch.setattr(
        "scripts.tools.compare_dense_models.create_query_translator",
        lambda settings: FakeTranslator(),
    )
    translator = LazySettingsQueryTranslator()

    assert calls == []
    assert translator.translate("吉他调音器") == "guitar tuner"
    assert calls == ["吉他调音器"]
