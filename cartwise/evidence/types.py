"""Types for evidence retrieval service orchestration."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from cartwise.recommendation.types import Diagnostic, RecommendedCandidate

if TYPE_CHECKING:
    from cartwise.evidence.rag import ProductExplanation

DEFAULT_REVIEW_EMBEDDING_MODEL = "intfloat/e5-small-v2"


def evidence_model_slug(model_name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", model_name).strip("_").lower()
    if not slug:
        raise ValueError("model name must contain at least one alphanumeric character")
    return slug


def evidence_collection_name(scope: str, model_name: str) -> str:
    return f"cartwise_review_evidence_{scope}_{evidence_model_slug(model_name)}"


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    parent_asin: str
    review_id: str
    chunk_id: str
    rating: float | None
    text: str | None
    chunk_text: str | None
    score: float | None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EvidenceRequest:
    query: str
    english_query: str
    candidates: Sequence[RecommendedCandidate]


@dataclass(frozen=True, slots=True)
class EvidenceResult:
    explanations: Sequence["ProductExplanation"]
    evidence_by_product: Mapping[str, Sequence[EvidenceItem]]
    diagnostics: Sequence[Diagnostic] = field(default_factory=tuple)
