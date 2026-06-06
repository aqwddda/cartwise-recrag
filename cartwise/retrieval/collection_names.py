"""Lightweight Qdrant collection naming helpers for retrieval indexes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DenseModelSpec:
    key: str
    model_name: str
    collection_suffix: str


PRODUCT_DENSE_MODEL_SPECS = {
    "e5": DenseModelSpec(
        key="e5",
        model_name="intfloat/e5-small-v2",
        collection_suffix="e5_small_v2",
    ),
    "blair": DenseModelSpec(
        key="blair",
        model_name="hyp1231/blair-roberta-base",
        collection_suffix="blair_roberta_base",
    ),
}


def product_collection_name(scope: str, model_key: str) -> str:
    try:
        suffix = PRODUCT_DENSE_MODEL_SPECS[model_key].collection_suffix
    except KeyError as error:
        raise ValueError(f"unsupported dense model: {model_key}") from error
    return f"cartwise_products_{scope}_{suffix}"
