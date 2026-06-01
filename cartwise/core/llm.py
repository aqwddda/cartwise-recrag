"""Minimal OpenAI-compatible LLM adapter used for search query translation."""

from __future__ import annotations

import re
from typing import Any, Protocol

import httpx
from openai import OpenAI, OpenAIError

from cartwise.core.config import Settings


CHINESE_CHARACTER_PATTERN = re.compile(r"[\u4e00-\u9fff]")
TRANSLATION_PROMPT = """Translate the following shopping search query into English.
Return only the translation without explanation:
{query}"""


class QueryTranslationError(RuntimeError):
    """Raised when a Chinese query cannot be translated for English retrieval."""


class QueryTranslator(Protocol):
    """Replaceable translation interface expanded by the later LLM stage."""

    def translate(self, query: str) -> str: ...


def contains_chinese_characters(query: str) -> bool:
    return CHINESE_CHARACTER_PATTERN.search(query) is not None


def build_translation_prompt(query: str) -> str:
    return TRANSLATION_PROMPT.format(query=query)


class OpenAICompatibleQueryTranslator:
    """Translate shopping queries through a configured chat completions API."""

    def __init__(self, client: Any, *, model: str) -> None:
        self.client = client
        self.model = model

    def translate(self, query: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": build_translation_prompt(query),
                    }
                ],
                temperature=0,
            )
            translation = response.choices[0].message.content
        except (OpenAIError, IndexError, AttributeError, TypeError) as error:
            raise QueryTranslationError("LLM query translation failed") from error
        if not isinstance(translation, str) or not translation.strip():
            raise QueryTranslationError("LLM query translation returned empty text")
        return translation.strip()


def create_query_translator(settings: Settings) -> OpenAICompatibleQueryTranslator:
    """Create the current OpenAI-compatible adapter from application settings."""

    if settings.llm_api_key is None:
        raise QueryTranslationError("LLM query translation is not configured")
    http_client = httpx.Client(
        proxy=settings.external_https_proxy,
        trust_env=False,
    )
    client = OpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        timeout=settings.llm_timeout_seconds,
        http_client=http_client,
    )
    return OpenAICompatibleQueryTranslator(client, model=settings.llm_model)


def prepare_search_query(
    query: str,
    *,
    translator: QueryTranslator | None = None,
) -> str:
    """Translate Chinese input and leave English retrieval input unchanged."""

    normalized = query.strip()
    if not normalized:
        raise ValueError("search query must not be empty")
    if not contains_chinese_characters(normalized):
        return normalized
    if translator is None:
        raise QueryTranslationError(
            "Chinese search query requires a configured LLM translator"
        )
    translation = translator.translate(normalized).strip()
    if not translation:
        raise QueryTranslationError("LLM query translation returned empty text")
    return translation
