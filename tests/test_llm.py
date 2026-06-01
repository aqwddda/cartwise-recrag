from __future__ import annotations

from types import SimpleNamespace

import pytest

from cartwise.core.llm import (
    OpenAICompatibleQueryTranslator,
    QueryTranslationError,
    build_translation_prompt,
    contains_chinese_characters,
    prepare_search_query,
)
from cartwise.core.config import Settings


class FakeTranslator:
    def __init__(self, translation: str) -> None:
        self.translation = translation
        self.queries: list[str] = []

    def translate(self, query: str) -> str:
        self.queries.append(query)
        return self.translation


def test_chinese_detection_uses_cjk_characters() -> None:
    assert contains_chinese_characters("适合初学者的吉他调音器")
    assert not contains_chinese_characters("guitar tuner for beginners")


def test_english_query_bypasses_translation() -> None:
    translator = FakeTranslator("unused")

    assert prepare_search_query("  guitar tuner  ", translator=translator) == (
        "guitar tuner"
    )
    assert translator.queries == []


def test_chinese_query_requires_translation() -> None:
    with pytest.raises(QueryTranslationError, match="requires"):
        prepare_search_query("吉他调音器")


def test_chinese_query_uses_translated_english_text() -> None:
    translator = FakeTranslator(" guitar tuner for beginners ")

    assert prepare_search_query("适合初学者的吉他调音器", translator=translator) == (
        "guitar tuner for beginners"
    )


def test_empty_translation_is_rejected() -> None:
    with pytest.raises(QueryTranslationError, match="empty"):
        prepare_search_query("吉他", translator=FakeTranslator(" "))


def test_openai_compatible_adapter_uses_fixed_prompt() -> None:
    request: dict[str, object] = {}

    class FakeCompletions:
        def create(self, **kwargs):
            request.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(message=SimpleNamespace(content="guitar tuner"))
                ]
            )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions()),
    )
    translator = OpenAICompatibleQueryTranslator(client, model="chat-model")

    assert translator.translate("吉他调音器") == "guitar tuner"
    assert request == {
        "model": "chat-model",
        "messages": [
            {
                "role": "user",
                "content": build_translation_prompt("吉他调音器"),
            }
        ],
        "temperature": 0,
    }


def test_google_key_uses_the_openai_compatible_gemini_fallback() -> None:
    google_base_url = Settings.model_fields["google_base_url"].default
    settings = Settings(
        _env_file=None,
        deepseek_api_key=None,
        google_api_key="google-key",
        google_base_url=google_base_url,
    )

    assert settings.llm_is_configured
    assert settings.llm_api_key == "google-key"
    assert settings.llm_base_url == google_base_url
    assert settings.llm_model == "gemini-2.5-flash"
