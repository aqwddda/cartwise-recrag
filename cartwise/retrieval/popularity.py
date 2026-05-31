"""Popularity recommendation baseline and offline ranking metrics."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


Interaction = Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class RankingMetrics:
    """Mean ranking metrics across users with evaluation interactions."""

    users: int
    recall: float
    ndcg: float
    hit_rate: float


def _group_items_by_user(interactions: Iterable[Interaction]) -> dict[str, set[str]]:
    items_by_user: dict[str, set[str]] = defaultdict(set)
    for interaction in interactions:
        items_by_user[interaction["user_id"]].add(interaction["parent_asin"])
    return dict(items_by_user)


def load_interactions(path: Path) -> list[dict[str, str]]:
    """Load only the columns required by recommenders and offline evaluation."""

    return pq.read_table(path, columns=["user_id", "parent_asin"]).to_pylist()


class PopularityRecommender:
    """Rank training items by interaction count and exclude user history."""

    def __init__(self, training_interactions: Iterable[Interaction]) -> None:
        self.item_counts: Counter[str] = Counter()
        interacted_items_by_user: dict[str, set[str]] = defaultdict(set)
        for interaction in training_interactions:
            user_id = interaction["user_id"]
            parent_asin = interaction["parent_asin"]
            self.item_counts[parent_asin] += 1
            interacted_items_by_user[user_id].add(parent_asin)
        self.interacted_items_by_user = dict(interacted_items_by_user)
        self.ranked_items = sorted(
            self.item_counts,
            key=lambda parent_asin: (-self.item_counts[parent_asin], parent_asin),
        )

    @classmethod
    def from_parquet(cls, path: Path) -> PopularityRecommender:
        return cls(load_interactions(path))

    def recommend(
        self,
        user_id: str,
        *,
        k: int = 10,
        excluded_items: Iterable[str] = (),
    ) -> list[str]:
        if k < 0:
            raise ValueError("k must be non-negative")
        if k == 0:
            return []

        excluded = set(excluded_items)
        excluded.update(self.interacted_items_by_user.get(user_id, ()))
        recommendations: list[str] = []
        for parent_asin in self.ranked_items:
            if parent_asin in excluded:
                continue
            recommendations.append(parent_asin)
            if len(recommendations) == k:
                break
        return recommendations


def _discounted_gain(recommendations: list[str], relevant_items: set[str]) -> float:
    return sum(
        1.0 / math.log2(rank + 2)
        for rank, parent_asin in enumerate(recommendations)
        if parent_asin in relevant_items
    )


def evaluate_recommender(
    recommender: PopularityRecommender,
    target_interactions: Iterable[Interaction],
    *,
    k: int = 10,
    additional_history: Iterable[Interaction] = (),
) -> RankingMetrics:
    """Evaluate recommendations while excluding interactions known before the split."""

    if k <= 0:
        raise ValueError("k must be greater than zero")

    targets_by_user = _group_items_by_user(target_interactions)
    history_by_user = _group_items_by_user(additional_history)
    if not targets_by_user:
        return RankingMetrics(users=0, recall=0.0, ndcg=0.0, hit_rate=0.0)

    recall = 0.0
    ndcg = 0.0
    hit_rate = 0.0
    for user_id, relevant_items in targets_by_user.items():
        recommendations = recommender.recommend(
            user_id,
            k=k,
            excluded_items=history_by_user.get(user_id, ()),
        )
        hits = relevant_items.intersection(recommendations)
        recall += len(hits) / len(relevant_items)
        hit_rate += float(bool(hits))
        ideal_gain = sum(
            1.0 / math.log2(rank + 2)
            for rank in range(min(len(relevant_items), k))
        )
        ndcg += _discounted_gain(recommendations, relevant_items) / ideal_gain

    users = len(targets_by_user)
    return RankingMetrics(
        users=users,
        recall=recall / users,
        ndcg=ndcg / users,
        hit_rate=hit_rate / users,
    )
