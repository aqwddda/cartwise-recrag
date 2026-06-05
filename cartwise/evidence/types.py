"""Types for evidence retrieval service orchestration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from cartwise.evidence.rag import ProductExplanation
from cartwise.recommendation.types import Diagnostic, RecommendedCandidate


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
    explanations: Sequence[ProductExplanation]
    evidence_by_product: Mapping[str, Sequence[EvidenceItem]]
    diagnostics: Sequence[Diagnostic] = field(default_factory=tuple)
