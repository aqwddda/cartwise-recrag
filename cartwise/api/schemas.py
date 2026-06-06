"""HTTP request and response schemas for the CartWise API."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from cartwise.application.types import (
    ApplicationRecommendation,
    ApplicationRecommendationRequest,
    ApplicationRecommendationResult,
)
from cartwise.recommendation.types import Diagnostic

DEFAULT_TOP_K = 10
MAX_TOP_K = 50


class RecommendRequest(BaseModel):
    """External API request for a single recommendation turn."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1)
    user_id: str | None = None
    top_k: int | None = Field(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        query = value.strip()
        if not query:
            raise ValueError("query must not be blank")
        return query

    @field_validator("user_id")
    @classmethod
    def normalize_user_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator("top_k", mode="before")
    @classmethod
    def default_top_k_when_null(cls, value: Any) -> Any:
        return DEFAULT_TOP_K if value is None else value

    def to_application_request(self) -> ApplicationRecommendationRequest:
        return ApplicationRecommendationRequest(
            query=self.query,
            user_id=self.user_id,
            top_k=self.top_k,
        )


class DiagnosticResponse(BaseModel):
    component: str
    error_type: str
    message: str
    recoverable: bool


class EvidenceResponse(BaseModel):
    review_id: str | None = None
    chunk_id: str | None = None
    rating: float | None = None
    text: str | None = None
    chunk_text: str | None = None
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RecommendationItemResponse(BaseModel):
    product_id: str
    parent_asin: str
    title: str | None = None
    brand: str | None = None
    price: float | None = None
    rank: int
    fusion_score: float
    sources: list[str]
    source_ranks: dict[str, int]
    source_scores: dict[str, Any]
    reason: str
    potential_cons: str
    fallback: bool
    evidence: list[EvidenceResponse]


class RecommendationResponse(BaseModel):
    query: str
    search_query: str
    known_user: bool
    applied_constraints: dict[str, Any]
    results: list[RecommendationItemResponse]
    diagnostics: list[DiagnosticResponse]
    latency_ms: int


class LiveHealthResponse(BaseModel):
    status: str


class ReadyHealthResponse(BaseModel):
    status: str
    application_service: str
    resources: dict[str, str]


def recommendation_response_from_result(
    result: ApplicationRecommendationResult,
    *,
    latency_ms: int,
) -> RecommendationResponse:
    """Convert internal application-service output to the stable HTTP shape."""

    return RecommendationResponse(
        query=result.query,
        search_query=result.search_query,
        known_user=result.known_user,
        applied_constraints=dict(result.applied_constraints),
        results=[
            _recommendation_item_response(recommendation)
            for recommendation in result.recommendations
        ],
        diagnostics=[
            _diagnostic_response(diagnostic)
            for diagnostic in result.diagnostics
        ],
        latency_ms=latency_ms,
    )


def _recommendation_item_response(
    recommendation: ApplicationRecommendation,
) -> RecommendationItemResponse:
    return RecommendationItemResponse(
        product_id=recommendation.parent_asin,
        parent_asin=recommendation.parent_asin,
        title=recommendation.title,
        brand=recommendation.brand,
        price=recommendation.price,
        rank=recommendation.rank,
        fusion_score=recommendation.fusion_score,
        sources=list(recommendation.sources),
        source_ranks=dict(recommendation.source_ranks),
        source_scores=dict(recommendation.source_scores),
        reason=recommendation.reason,
        potential_cons=recommendation.potential_cons,
        fallback=recommendation.fallback,
        evidence=[
            _evidence_response(evidence)
            for evidence in recommendation.evidence
        ],
    )


def _evidence_response(evidence: Mapping[str, Any]) -> EvidenceResponse:
    return EvidenceResponse(
        review_id=_optional_string(evidence.get("review_id")),
        chunk_id=_optional_string(evidence.get("chunk_id")),
        rating=evidence.get("rating"),
        text=_optional_string(evidence.get("text")),
        chunk_text=_optional_string(evidence.get("chunk_text")),
        score=evidence.get("score"),
        metadata=dict(evidence.get("metadata") or {}),
    )


def _diagnostic_response(diagnostic: Diagnostic) -> DiagnosticResponse:
    return DiagnosticResponse(
        component=diagnostic.component,
        error_type=diagnostic.error_type,
        message=diagnostic.message,
        recoverable=diagnostic.recoverable,
    )


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
