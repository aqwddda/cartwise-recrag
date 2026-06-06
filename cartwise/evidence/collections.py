"""Lightweight Qdrant collection naming helpers for review evidence indexes."""

from __future__ import annotations

import re


DEFAULT_REVIEW_EMBEDDING_MODEL = "intfloat/e5-small-v2"


def evidence_model_slug(model_name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", model_name).strip("_").lower()
    if not slug:
        raise ValueError("model name must contain at least one alphanumeric character")
    return slug


def evidence_collection_name(scope: str, model_name: str) -> str:
    return f"cartwise_review_evidence_{scope}_{evidence_model_slug(model_name)}"
