from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from cartwise.retrieval.lightgcn import (
    LightGCNConfig,
    LightGCNRecommender,
    evaluate_lightgcn_recommender,
    resolve_device,
    train_lightgcn,
)


TRAIN_INTERACTIONS = [
    {"user_id": "U1", "parent_asin": "P1"},
    {"user_id": "U1", "parent_asin": "P2"},
    {"user_id": "U2", "parent_asin": "P2"},
    {"user_id": "U2", "parent_asin": "P3"},
    {"user_id": "U3", "parent_asin": "P3"},
    {"user_id": "U3", "parent_asin": "P4"},
    {"user_id": "U4", "parent_asin": "P4"},
    {"user_id": "U4", "parent_asin": "P5"},
]
TINY_CONFIG = LightGCNConfig(embedding_dim=4, num_layers=1)


def build_cpu_recommender() -> LightGCNRecommender:
    return LightGCNRecommender.from_training_interactions(
        TRAIN_INTERACTIONS,
        config=TINY_CONFIG,
        device="cpu",
    )


def test_training_graph_contains_only_supplied_interactions() -> None:
    recommender = build_cpu_recommender()

    assert recommender.user_to_index == {"U1": 0, "U2": 1, "U3": 2, "U4": 3}
    assert recommender.item_to_index == {
        "P1": 0,
        "P2": 1,
        "P3": 2,
        "P4": 3,
        "P5": 4,
    }
    assert recommender.edge_index.shape == (2, len(TRAIN_INTERACTIONS) * 2)


def test_recommend_excludes_history_and_explicit_items() -> None:
    recommender = build_cpu_recommender()

    recommendations = recommender.recommend(
        "U1",
        k=10,
        excluded_items={"P3"},
    )

    assert set(recommendations) == {"P4", "P5"}


def test_unknown_user_returns_empty_list() -> None:
    recommender = build_cpu_recommender()

    assert recommender.recommend("unknown-user") == []


def test_save_and_load_preserve_recommendations(tmp_path: Path) -> None:
    recommender = build_cpu_recommender()
    expected = recommender.recommend("U1", k=3)
    path = tmp_path / "lightgcn.pt"

    recommender.save(path)
    loaded = LightGCNRecommender.load(path, device="cpu")

    assert loaded.recommend("U1", k=3) == expected
    assert loaded.interacted_items_by_user == recommender.interacted_items_by_user


def test_tiny_graph_trains_on_cpu_with_bpr_loss() -> None:
    reported_losses: list[tuple[int, float]] = []
    recommender, losses = train_lightgcn(
        TRAIN_INTERACTIONS,
        config=TINY_CONFIG,
        epochs=2,
        device="cpu",
        epoch_callback=lambda epoch, loss: reported_losses.append((epoch, loss)),
    )

    assert len(losses) == 2
    assert all(math.isfinite(loss) for loss in losses)
    assert reported_losses == [(1, losses[0]), (2, losses[1])]
    assert set(recommender.recommend("U1", k=10)).isdisjoint({"P1", "P2"})


def test_batch_evaluation_uses_shared_metric_contract() -> None:
    recommender = build_cpu_recommender()

    metrics = evaluate_lightgcn_recommender(
        recommender,
        [{"user_id": "U1", "parent_asin": "P4"}],
        k=2,
        additional_history=[{"user_id": "U1", "parent_asin": "P3"}],
    )

    assert metrics.users == 1
    assert metrics.recall == 1.0
    assert metrics.hit_rate == 1.0
    assert 0.0 < metrics.ndcg <= 1.0


def test_cuda_request_fails_instead_of_falling_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA device requested"):
        resolve_device("cuda")
