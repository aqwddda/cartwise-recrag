"""Types for the top-level recommendation application service."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from cartwise.recommendation.types import Diagnostic, RecommendationResult
from cartwise.evidence.types import EvidenceResult


@dataclass(frozen=True, slots=True)
class ApplicationRecommendationRequest:
    query: str
    user_id: str | None = None
    top_k: int | None = None


@dataclass(frozen=True, slots=True)
class ApplicationRecommendation:
    parent_asin: str
    title: str | None
    brand: str | None
    price: float | None
    rank: int
    fusion_score: float
    sources: Sequence[str]
    source_ranks: Mapping[str, int]
    source_scores: Mapping[str, Any]
    reason: str
    potential_cons: str
    evidence: Sequence[Mapping[str, Any]]
    fallback: bool


@dataclass(frozen=True, slots=True)
class ApplicationRecommendationResult:
    query: str
    search_query: str
    known_user: bool
    applied_constraints: Mapping[str, Any]
    recommendations: Sequence[ApplicationRecommendation]
    recommendation_result: RecommendationResult
    evidence_result: EvidenceResult
    diagnostics: Sequence[Diagnostic] = field(default_factory=tuple)
