from __future__ import annotations

from types import SimpleNamespace

import pytest

from cartwise.core.llm import (
    OpenAICompatibleQueryIntentParser,
    OpenAICompatibleQueryTranslator,
    QueryIntentError,
    QueryTranslationError,
    build_intent_prompt,
    build_translation_prompt,
    contains_chinese_characters,
    prepare_search_query,
    validate_query_intent,
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


def test_openai_compatible_intent_parser_requests_strict_json() -> None:
    request: dict[str, object] = {}

    class FakeCompletions:
        def create(self, **kwargs):
            request.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=(
                                '{"product_terms":["guitar tuner"],"min_price":null,'
                                '"max_price":50,"brands":[],"excluded_brands":'
                                '["Fender"],"color_tags":["black"],'
                                '"material_tags":["plastic"]}'
                            )
                        )
                    )
                ]
            )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions()),
    )
    parser = OpenAICompatibleQueryIntentParser(
        client,
        model="deepseek-chat",
    )

    intent = parser.parse("guitar tuner under $50 not Fender")

    assert intent.search_query == "guitar tuner under $50 not Fender"
    assert intent.product_terms == ("guitar tuner",)
    assert intent.filters.max_price == 50.0
    assert tuple(intent.filters.excluded_brands) == ("Fender",)
    assert tuple(intent.filters.color_tags) == ("black",)
    assert tuple(intent.filters.material_tags) == ("plastic",)
    assert request == {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "user",
                "content": build_intent_prompt(
                    "guitar tuner under $50 not Fender",
                ),
            }
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "extra_body": {"thinking": {"type": "disabled"}},
    }


def test_validate_query_intent_reuses_translation_for_chinese_search_query() -> None:
    translator = FakeTranslator("guitar tuner for beginners")

    intent = validate_query_intent(
        "适合初学者的吉他调音器",
        {
            "search_query": "hallucinated rewrite",
            "product_terms": ["吉他调音器"],
            "min_price": None,
            "max_price": None,
            "brands": [],
            "excluded_brands": [],
            "color_tags": [],
            "material_tags": [],
        },
        translator=translator,
    )

    assert intent.search_query == "guitar tuner for beginners"
    assert intent.product_terms == ("吉他调音器",)
    assert translator.queries == ["适合初学者的吉他调音器"]


def test_validate_query_intent_ignores_llm_rewrite_for_english_search_query() -> None:
    intent = validate_query_intent(
        "clip-on guitar tuner under $30 not Fender",
        {
            "search_query": "guitar tuner",
            "product_terms": ["clip-on guitar tuner"],
            "max_price": 30,
            "excluded_brands": ["Fender"],
        },
    )

    assert intent.search_query == "clip-on guitar tuner under $30 not Fender"
    assert intent.product_terms == ("clip-on guitar tuner",)
    assert intent.filters.max_price == 30.0
    assert tuple(intent.filters.excluded_brands) == ("Fender",)


def test_validate_query_intent_keeps_raw_llm_terms_without_vocabulary_matching() -> None:
    intent = validate_query_intent(
        "black wood guitar strap not fender",
        {
            "product_terms": ["guitar strap", "Imaginary Category"],
            "brands": ["Acme"],
            "excluded_brands": ["Fndr", "Unknown"],
            "color_tags": [" black "],
            "material_tags": ["wood"],
        },
    )

    assert intent.product_terms == ("guitar strap", "Imaginary Category")
    assert tuple(intent.filters.brands) == ("Acme",)
    assert tuple(intent.filters.excluded_brands) == ("Fndr", "Unknown")
    assert tuple(intent.filters.color_tags) == ("black",)
    assert tuple(intent.filters.material_tags) == ("wood",)


def test_validate_query_intent_uses_llm_price_without_regex_override() -> None:
    intent = validate_query_intent(
        "microphone stand between $20 and $40",
        {"min_price": 1, "max_price": 999},
    )

    assert intent.filters.min_price == 1.0
    assert intent.filters.max_price == 999.0


def test_invalid_intent_schema_is_rejected() -> None:
    with pytest.raises(QueryIntentError, match="schema validation"):
        validate_query_intent(
            "guitar tuner",
            {"min_price": 50, "max_price": 10},
        )


def test_invalid_intent_json_is_rejected() -> None:
    class FakeCompletions:
        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="not json"))]
            )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions()),
    )
    parser = OpenAICompatibleQueryIntentParser(client, model="deepseek-chat")

    with pytest.raises(QueryIntentError, match="invalid JSON"):
        parser.parse("guitar tuner")


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
