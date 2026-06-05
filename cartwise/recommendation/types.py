"""Types shared by recommendation orchestration and callers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from cartwise.query.types import FilterConstraints
from cartwise.retrieval.fusion import FusionOutput


@dataclass(frozen=True, slots=True)
class Diagnostic:
    component: str
    error_type: str
    message: str
    recoverable: bool


@dataclass(frozen=True, slots=True)
class RecommendedCandidate:
    parent_asin: str
    rank: int
    fusion_score: float
    sources: tuple[str, ...]
    source_ranks: Mapping[str, int]
    source_scores: Mapping[str, Any]
    item: Mapping[str, Any]
    retrieval_query: Any = None
    document: Any = None


def recommended_candidate_from_mapping(
    record: Mapping[str, Any],
) -> RecommendedCandidate:
    return RecommendedCandidate(
        parent_asin=str(record["parent_asin"]),
        rank=int(record.get("rank", 0)),
        fusion_score=float(record.get("fusion_score", record.get("score", 0.0))),
        sources=tuple(record.get("sources", ())),
        source_ranks=dict(record.get("source_ranks", {})),
        source_scores=dict(record.get("source_scores", {})),
        item=dict(record.get("item", {"parent_asin": record["parent_asin"]})),
        retrieval_query=record.get("retrieval_query"),
        document=record.get("document"),
    )


@dataclass(frozen=True, slots=True)
class RecommendationRequest:
    query: str
    user_id: str | None = None
    top_k: int | None = None
    mode: Literal["fusion", "smoke_search_only"] = "fusion"


@dataclass(frozen=True, slots=True)
class RecommendationResult:
    query: str
    search_query: str
    known_user: bool
    intent: Mapping[str, Any]
    filter_constraints: FilterConstraints
    filter_constraints_payload: Mapping[str, Any]
    candidates_by_channel: Mapping[str, Sequence[Mapping[str, Any]]]
    fusion_output: FusionOutput
    final_candidates: Sequence[Mapping[str, Any]]
    diagnostics: Sequence[Diagnostic] = field(default_factory=tuple)
