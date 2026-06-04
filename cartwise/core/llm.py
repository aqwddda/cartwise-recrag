"""OpenAI-compatible LLM adapters for query translation and intent parsing."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import httpx
from openai import OpenAI, OpenAIError
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from cartwise.core.config import Settings
from cartwise.retrieval.filters import FilterConstraints

CHINESE_CHARACTER_PATTERN = re.compile(r"[\u4e00-\u9fff]")
TRANSLATION_PROMPT = """Translate the following shopping search query into English.
Return only the translation without explanation:
{query}"""
INTENT_PROMPT_TEMPLATE = """Extract explicit shopping intent from the user query.

Return exactly this JSON object shape:
{{
  "product_terms": [],
  "brands": [],
  "excluded_brands": [],
  "min_price": null,
  "max_price": null,
  "color_tags": [],
  "material_tags": []
}}

Rules:
- Do not recommend products or explain.
- Do not rewrite or translate the retrieval query.
- product_terms are the core product words the user wants to buy.
- brands are explicitly requested brand names.
- excluded_brands are explicitly rejected brand names.
- Use min_price and max_price only for explicit numeric price constraints.
- Do not convert vague words such as cheap, affordable, beginner, or premium into prices.
- color_tags and material_tags are explicit colors or materials from the query.
- If a list field is uncertain, use [].
- If a price field is uncertain, use null.
- If a list field contains Chinese text, translate it into English.
- product_terms should contain canonical product types, not descriptive modifiers.
- Remove size, quality, portability, skill-level, or subjective descriptors unless they fundamentally change the product category.

User query:
{query}"""


class QueryTranslationError(RuntimeError):
    """Raised when a Chinese query cannot be translated for English retrieval."""


class QueryIntentError(RuntimeError):
    """Raised when a query intent cannot be parsed into a valid structure."""


class QueryTranslator(Protocol):
    """Replaceable translation interface expanded by the later LLM stage."""

    def translate(self, query: str) -> str: ...


class QueryIntentParser(Protocol):
    """Replaceable query intent parser for the stage-seven recommendation flow."""

    def parse(self, query: str) -> "ParsedQueryIntent": ...


@dataclass(frozen=True, slots=True)
class ParsedQueryIntent:
    """Validated query intent used by retrieval and hard-filter stages."""

    search_query: str
    product_terms: tuple[str, ...]
    filters: FilterConstraints


class QueryIntentPayload(BaseModel):
    """Pydantic contract for raw LLM query intent JSON."""

    model_config = ConfigDict(extra="ignore")

    product_terms: tuple[str, ...] = Field(default_factory=tuple)
    brands: tuple[str, ...] = Field(default_factory=tuple)
    excluded_brands: tuple[str, ...] = Field(default_factory=tuple)
    min_price: float | None = None
    max_price: float | None = None
    color_tags: tuple[str, ...] = Field(default_factory=tuple)
    material_tags: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator(
        "product_terms",
        "brands",
        "excluded_brands",
        "color_tags",
        "material_tags",
        mode="after",
    )
    @classmethod
    def _drop_empty_strings(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(value.strip() for value in values if value.strip())

    @field_validator("min_price", "max_price", mode="after")
    @classmethod
    def _validate_price(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if not math.isfinite(value) or value < 0:
            raise ValueError("price must be a non-negative finite number")
        return value

    @model_validator(mode="after")
    def _validate_price_range(self) -> "QueryIntentPayload":
        if (
            self.min_price is not None
            and self.max_price is not None
            and self.min_price > self.max_price
        ):
            raise ValueError("min_price must not exceed max_price")
        return self


def contains_chinese_characters(query: str) -> bool:
    return CHINESE_CHARACTER_PATTERN.search(query) is not None


def build_translation_prompt(query: str) -> str:
    return TRANSLATION_PROMPT.format(query=query)


def build_intent_prompt(query: str) -> str:
    return INTENT_PROMPT_TEMPLATE.format(query=query)


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


def create_query_intent_parser(
    settings: Settings,
    *,
    translator: QueryTranslator | None = None,
) -> OpenAICompatibleQueryIntentParser:
    """Create the stage-seven OpenAI-compatible intent parser from settings."""

    if settings.llm_api_key is None:
        raise QueryIntentError("LLM query intent parsing is not configured")
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
    return OpenAICompatibleQueryIntentParser(
        client,
        model=settings.llm_model,
        translator=translator,
    )


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


class OpenAICompatibleQueryIntentParser:
    """Parse shopping query intent through a configured chat completions API."""

    def __init__(
        self,
        client: Any,
        *,
        model: str,
        translator: QueryTranslator | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.translator = translator

    def parse(self, query: str) -> ParsedQueryIntent:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be empty")
        prompt = build_intent_prompt(normalized_query)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"},
                extra_body={"thinking": {"type": "disabled"}},
            )
            content = response.choices[0].message.content
        except (OpenAIError, IndexError, AttributeError, TypeError) as error:
            raise QueryIntentError("LLM query intent parsing failed") from error
        if not isinstance(content, str) or not content.strip():
            raise QueryIntentError("LLM query intent parsing returned empty JSON")
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as error:
            raise QueryIntentError(
                "LLM query intent parsing returned invalid JSON"
            ) from error
        return validate_query_intent(
            normalized_query,
            payload,
            translator=self.translator,
        )


def validate_query_intent(
    query: str,
    payload: Mapping[str, Any],
    *,
    translator: QueryTranslator | None = None,
) -> ParsedQueryIntent:
    """Validate LLM JSON and convert raw fields into filter constraints."""

    search_query = _resolve_intent_search_query(
        query,
        payload,
        translator=translator,
    )
    try:
        intent = QueryIntentPayload.model_validate(payload)
    except ValidationError as error:
        raise QueryIntentError(
            "LLM query intent JSON failed schema validation"
        ) from error
    constraints = FilterConstraints(
        min_price=intent.min_price,
        max_price=intent.max_price,
        brands=intent.brands,
        excluded_brands=intent.excluded_brands,
        color_tags=intent.color_tags,
        material_tags=intent.material_tags,
    )
    return ParsedQueryIntent(
        search_query=search_query,
        product_terms=intent.product_terms,
        filters=constraints,
    )


def _resolve_intent_search_query(
    query: str,
    payload: Mapping[str, Any],
    *,
    translator: QueryTranslator | None,
) -> str:
    del payload
    return prepare_search_query(query, translator=translator)
