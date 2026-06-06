"""Stage-eight review evidence retrieval and grounded explanation generation."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


EXPLANATION_PROMPT = PromptTemplate.from_template(
    """Generate grounded Chinese shopping explanations from review evidence.

Return only a JSON object with this shape:
{{
  "items": [
    {{
      "parent_asin": "...",
      "reason": "...",
      "potential_cons": "...",
      "cited_review_ids": ["..."]
    }}
  ]
}}

Rules:
- Use only the candidate products and review evidence below.
- reason and potential_cons must be written in Chinese.
- Do not add products outside the candidate list.
- Do not invent prices, brands, attributes, review content, or review IDs.
- cited_review_ids must cite review_id values, never chunk_id values.
- Each cited review_id must belong to the same parent_asin.

English query:
{query}

Candidates and evidence:
{evidence}
"""
)


@dataclass(frozen=True, slots=True)
class EvidenceRagConfig:
    initial_chunk_k: int = 10
    final_review_k: int = 5
    max_candidate_chunk_k: int = 20


@dataclass(frozen=True, slots=True)
class ReviewEvidence:
    parent_asin: str
    review_id: str
    chunk_id: str
    rating: float | None
    title: str | None
    text: str | None
    chunk_text: str
    helpful_vote: int | None = None
    verified_purchase: bool | None = None
    timestamp: int | None = None
    score: float | None = None


@dataclass(frozen=True, slots=True)
class ProductExplanation:
    parent_asin: str
    reason: str
    potential_cons: str
    cited_review_ids: tuple[str, ...]
    evidence: tuple[ReviewEvidence, ...]
    fallback: bool


class ReviewEvidenceRetriever(Protocol):
    def search(
        self,
        query: str,
        *,
        parent_asin: str,
        k: int,
        rating_lte: float | None = None,
    ) -> list[Mapping[str, Any]]: ...


class ExplanationGenerator(Protocol):
    def generate(self, prompt: str) -> str: ...


class ReviewEvidenceEncoder(Protocol):
    def encode_query(self, query: str) -> np.ndarray: ...


class EvidenceRagError(RuntimeError):
    """Raised when retrieved evidence or generated explanations fail validation."""


class ExplanationItemPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    parent_asin: str
    reason: str
    potential_cons: str
    cited_review_ids: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator("parent_asin", "reason", "potential_cons", mode="after")
    @classmethod
    def _strip_required_text(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("field must be a non-empty string")
        return value.strip()

    @field_validator("cited_review_ids", mode="after")
    @classmethod
    def _strip_review_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(value.strip() for value in values if value.strip())


class ExplanationPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    items: tuple[ExplanationItemPayload, ...]


class OpenAICompatibleExplanationGenerator:
    """Generate explanation JSON through an OpenAI-compatible chat client."""

    def __init__(self, client: Any, *, model: str) -> None:
        self.client = client
        self.model = model

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
            extra_body={"thinking": {"type": "disabled"}},
        )
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise EvidenceRagError("LLM explanation generation returned empty JSON")
        return content


class QdrantReviewEvidenceRetriever:
    """Retrieve review evidence from the stage-eight review Qdrant collection."""

    def __init__(
        self,
        client: Any,
        *,
        collection: str,
        encoder: ReviewEvidenceEncoder,
    ) -> None:
        self.client = client
        self.collection = collection
        self.encoder = encoder

    def search(
        self,
        query: str,
        *,
        parent_asin: str,
        k: int,
        rating_lte: float | None = None,
    ) -> list[Mapping[str, Any]]:
        if k <= 0:
            raise ValueError("k must be greater than zero")
        filters = [
            _field_condition("parent_asin", parent_asin),
        ]
        if rating_lte is not None:
            filters.append(
                {
                    "key": "rating",
                    "range": {"lte": rating_lte},
                }
            )
        response = self.client.query_points(
            collection_name=self.collection,
            query=self.encoder.encode_query(query).tolist(),
            limit=k,
            query_filter={"must": filters},
            with_payload=True,
        )
        results: list[Mapping[str, Any]] = []
        for point in response.points:
            payload = dict(point.payload or {})
            payload["score"] = point.score
            results.append(payload)
        return results


def explain_candidates(
    *,
    english_query: str,
    candidates: Sequence[Mapping[str, Any]],
    retriever: ReviewEvidenceRetriever,
    generator: ExplanationGenerator | None = None,
    config: EvidenceRagConfig = EvidenceRagConfig(),
) -> list[ProductExplanation]:
    """Retrieve review evidence for fusion candidates and return grounded explanations."""

    _validate_config(config)
    candidate_map = _candidate_map(candidates)
    try:
        evidence_by_product = {
            parent_asin: retrieve_product_evidence(
                english_query=english_query,
                candidate=candidate,
                retriever=retriever,
                config=config,
            )
            for parent_asin, candidate in candidate_map.items()
        }
    except (EvidenceRagError, ValueError, KeyError, TypeError):
        evidence_by_product = {parent_asin: () for parent_asin in candidate_map}
        return _fallback_explanations(candidate_map, evidence_by_product)
    if generator is None:
        return _fallback_explanations(candidate_map, evidence_by_product)
    try:
        prompt = build_explanation_prompt(
            english_query=english_query,
            candidates=list(candidate_map.values()),
            evidence_by_product=evidence_by_product,
        )
        payload = _parse_explanation_payload(generator.generate(prompt))
        return _validated_explanations(payload, candidate_map, evidence_by_product)
    except (EvidenceRagError, ValueError, KeyError, TypeError, ValidationError, json.JSONDecodeError):
        return _fallback_explanations(candidate_map, evidence_by_product)


def retrieve_product_evidence(
    *,
    english_query: str,
    candidate: Mapping[str, Any],
    retriever: ReviewEvidenceRetriever,
    config: EvidenceRagConfig = EvidenceRagConfig(),
) -> tuple[ReviewEvidence, ...]:
    parent_asin = _required_parent_asin(candidate)
    query = build_review_query(english_query, candidate)
    hits = list(
        retriever.search(
            query,
            parent_asin=parent_asin,
            k=config.initial_chunk_k,
        )
    )
    _validate_hits_scope(hits, parent_asin=parent_asin)
    if _distinct_review_count(hits) < config.final_review_k:
        expanded_hits = list(
            retriever.search(
                query,
                parent_asin=parent_asin,
                k=config.max_candidate_chunk_k,
            )
        )
        _validate_hits_scope(expanded_hits, parent_asin=parent_asin)
        hits = _merge_hits(
            hits,
            expanded_hits,
        )
    evidence = _select_final_evidence(
        hits,
        parent_asin=parent_asin,
        final_review_k=config.final_review_k,
    )
    if evidence and not any(_is_low_rating(entry.rating) for entry in evidence):
        evidence = _include_low_rating_evidence(
            evidence,
            retriever.search(
                query,
                parent_asin=parent_asin,
                k=config.max_candidate_chunk_k,
                rating_lte=3,
            ),
            parent_asin=parent_asin,
            final_review_k=config.final_review_k,
        )
    return tuple(evidence)


def build_review_query(english_query: str, candidate: Mapping[str, Any]) -> str:
    item = _candidate_item(candidate)
    fields = [
        english_query.strip(),
        str(item.get("title") or "").strip(),
        _render_categories(item.get("categories")),
    ]
    query = "\n".join(field for field in fields if field)
    if not query:
        raise ValueError("review evidence query must not be empty")
    return query


def build_explanation_prompt(
    *,
    english_query: str,
    candidates: Sequence[Mapping[str, Any]],
    evidence_by_product: Mapping[str, Sequence[ReviewEvidence]],
) -> str:
    evidence_json = json.dumps(
        [
            {
                "parent_asin": _required_parent_asin(candidate),
                "item": _candidate_item(candidate),
                "reviews": [
                    _evidence_payload(evidence)
                    for evidence in evidence_by_product.get(
                        _required_parent_asin(candidate),
                        (),
                    )
                ],
            }
            for candidate in candidates
        ],
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    )
    return EXPLANATION_PROMPT.format(query=english_query, evidence=evidence_json)


def template_explanation(
    candidate: Mapping[str, Any],
    evidence: Sequence[ReviewEvidence],
) -> ProductExplanation:
    parent_asin = _required_parent_asin(candidate)
    item = _candidate_item(candidate)
    title = str(item.get("title") or parent_asin)
    cited_review_ids = tuple(dict.fromkeys(entry.review_id for entry in evidence))
    if evidence:
        reason = f"{title} 符合当前需求；已检索到 {len(cited_review_ids)} 条可追溯评论证据。"
        low_rating = next((entry for entry in evidence if _is_low_rating(entry.rating)), None)
        if low_rating is None:
            potential_cons = "暂未检索到明确的中低评分评论，潜在缺点需要后续人工核查。"
        else:
            potential_cons = (
                f"部分评论评分较低（review_id: {low_rating.review_id}），"
                "建议关注该评论中提到的使用体验。"
            )
    else:
        reason = f"{title} 来自候选商品列表，但当前没有可用评论证据支持更详细解释。"
        potential_cons = "缺少可引用评论证据，暂不生成具体缺点。"
    return ProductExplanation(
        parent_asin=parent_asin,
        reason=reason,
        potential_cons=potential_cons,
        cited_review_ids=cited_review_ids,
        evidence=tuple(evidence),
        fallback=True,
    )


def _validate_config(config: EvidenceRagConfig) -> None:
    if config.initial_chunk_k <= 0:
        raise ValueError("initial_chunk_k must be greater than zero")
    if config.final_review_k <= 0:
        raise ValueError("final_review_k must be greater than zero")
    if config.max_candidate_chunk_k < config.initial_chunk_k:
        raise ValueError("max_candidate_chunk_k must be at least initial_chunk_k")


def _field_condition(key: str, value: Any) -> dict[str, Any]:
    return {"key": key, "match": {"value": value}}


def _candidate_map(
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    records: dict[str, Mapping[str, Any]] = {}
    for candidate in candidates:
        parent_asin = _required_parent_asin(candidate)
        records[parent_asin] = candidate
    return records


def _required_parent_asin(candidate: Mapping[str, Any]) -> str:
    parent_asin = str(candidate.get("parent_asin") or "").strip()
    if not parent_asin:
        raise ValueError("candidate parent_asin must not be empty")
    return parent_asin


def _candidate_item(candidate: Mapping[str, Any]) -> dict[str, Any]:
    item = candidate.get("item")
    return dict(item) if isinstance(item, Mapping) else {}


def _render_categories(value: Any) -> str:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ""
    return " | ".join(str(entry).strip() for entry in value if str(entry).strip())


def _distinct_review_count(hits: Sequence[Mapping[str, Any]]) -> int:
    return len({str(hit.get("review_id") or "").strip() for hit in hits if hit.get("review_id")})


def _merge_hits(
    first: Sequence[Mapping[str, Any]],
    second: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    merged: list[Mapping[str, Any]] = []
    seen_chunks: set[str] = set()
    for hit in [*first, *second]:
        chunk_id = str(hit.get("chunk_id") or "").strip()
        if not chunk_id or chunk_id in seen_chunks:
            continue
        seen_chunks.add(chunk_id)
        merged.append(hit)
    return merged


def _validate_hits_scope(
    hits: Sequence[Mapping[str, Any]],
    *,
    parent_asin: str,
) -> None:
    for hit in hits:
        if str(hit.get("parent_asin") or "").strip() != parent_asin:
            raise EvidenceRagError("review retrieval returned evidence outside candidate scope")


def _select_final_evidence(
    hits: Sequence[Mapping[str, Any]],
    *,
    parent_asin: str,
    final_review_k: int,
) -> list[ReviewEvidence]:
    accepted_review_ids: list[str] = []
    for hit in hits:
        evidence = _review_evidence(hit)
        if evidence.parent_asin != parent_asin:
            raise EvidenceRagError("review retrieval returned evidence outside candidate scope")
        if evidence.review_id not in accepted_review_ids:
            accepted_review_ids.append(evidence.review_id)
        if len(accepted_review_ids) == final_review_k:
            break
    accepted = set(accepted_review_ids)
    return [
        evidence
        for hit in hits
        if (evidence := _review_evidence(hit)).review_id in accepted
    ]


def _include_low_rating_evidence(
    evidence: Sequence[ReviewEvidence],
    low_rating_hits: Sequence[Mapping[str, Any]],
    *,
    parent_asin: str,
    final_review_k: int,
) -> list[ReviewEvidence]:
    low_rating = next(
        (
            _review_evidence(hit)
            for hit in low_rating_hits
            if _review_evidence(hit).review_id
            not in {entry.review_id for entry in evidence}
        ),
        None,
    )
    if low_rating is None:
        return list(evidence)
    if low_rating.parent_asin != parent_asin:
        raise EvidenceRagError("low-rating retrieval returned evidence outside candidate scope")
    by_review_id: dict[str, list[ReviewEvidence]] = {}
    for entry in evidence:
        by_review_id.setdefault(entry.review_id, []).append(entry)
    if len(by_review_id) < final_review_k:
        return [*evidence, low_rating]
    dropped_review_id = next(reversed(by_review_id))
    return [
        entry for entry in evidence if entry.review_id != dropped_review_id
    ] + [low_rating]


def _review_evidence(hit: Mapping[str, Any]) -> ReviewEvidence:
    parent_asin = str(hit.get("parent_asin") or "").strip()
    review_id = str(hit.get("review_id") or "").strip()
    chunk_id = str(hit.get("chunk_id") or "").strip()
    chunk_text = str(hit.get("chunk_text") or "").strip()
    if not parent_asin or not review_id or not chunk_id or not chunk_text:
        raise EvidenceRagError("review evidence payload is missing required fields")
    return ReviewEvidence(
        parent_asin=parent_asin,
        review_id=review_id,
        chunk_id=chunk_id,
        rating=_read_float(hit.get("rating")),
        title=_read_optional_text(hit.get("title")),
        text=_read_optional_text(hit.get("text")),
        chunk_text=chunk_text,
        helpful_vote=_read_optional_int(hit.get("helpful_vote")),
        verified_purchase=_read_optional_bool(hit.get("verified_purchase")),
        timestamp=_read_optional_int(hit.get("timestamp")),
        score=_read_float(hit.get("score")),
    )


def _read_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _read_optional_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _read_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _is_low_rating(rating: float | None) -> bool:
    return rating is not None and rating <= 3


def _evidence_payload(evidence: ReviewEvidence) -> dict[str, Any]:
    return {
        "review_id": evidence.review_id,
        "chunk_id": evidence.chunk_id,
        "rating": evidence.rating,
        "title": evidence.title,
        "chunk_text": evidence.chunk_text,
        "helpful_vote": evidence.helpful_vote,
        "verified_purchase": evidence.verified_purchase,
        "timestamp": evidence.timestamp,
    }


def _parse_explanation_payload(content: str) -> ExplanationPayload:
    return ExplanationPayload.model_validate(json.loads(content))


def _validated_explanations(
    payload: ExplanationPayload,
    candidate_map: Mapping[str, Mapping[str, Any]],
    evidence_by_product: Mapping[str, Sequence[ReviewEvidence]],
) -> list[ProductExplanation]:
    seen: set[str] = set()
    explanations: list[ProductExplanation] = []
    for item in payload.items:
        if item.parent_asin not in candidate_map:
            raise EvidenceRagError("LLM returned product outside candidate list")
        evidence = tuple(evidence_by_product.get(item.parent_asin, ()))
        allowed_review_ids = {entry.review_id for entry in evidence}
        cited_review_ids = tuple(dict.fromkeys(item.cited_review_ids))
        if any(review_id not in allowed_review_ids for review_id in cited_review_ids):
            raise EvidenceRagError("LLM cited a review outside retrieved evidence")
        explanations.append(
            ProductExplanation(
                parent_asin=item.parent_asin,
                reason=item.reason,
                potential_cons=item.potential_cons,
                cited_review_ids=cited_review_ids,
                evidence=evidence,
                fallback=False,
            )
        )
        seen.add(item.parent_asin)
    if seen != set(candidate_map):
        raise EvidenceRagError("LLM did not return exactly one item per candidate")
    return explanations


def _fallback_explanations(
    candidate_map: Mapping[str, Mapping[str, Any]],
    evidence_by_product: Mapping[str, Sequence[ReviewEvidence]],
) -> list[ProductExplanation]:
    return [
        template_explanation(candidate, evidence_by_product.get(parent_asin, ()))
        for parent_asin, candidate in candidate_map.items()
    ]
