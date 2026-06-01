"""PyTorch Geometric LightGCN training, persistence, and inference."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch_geometric.nn.models import LightGCN

from cartwise.retrieval.popularity import (
    Interaction,
    RankingMetrics,
    group_items_by_user,
    ranking_metrics_from_recommendations,
)


MODEL_FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True)
class LightGCNConfig:
    """Architecture parameters required to rebuild a saved model."""

    embedding_dim: int = 64
    num_layers: int = 3


def resolve_device(device: str | torch.device) -> torch.device:
    """Resolve a requested device without silently falling back from CUDA."""

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested, but torch.cuda.is_available() is False")
    return resolved


def _build_mappings(
    interactions: list[Interaction],
) -> tuple[dict[str, int], dict[str, int], dict[str, set[str]]]:
    users = sorted({interaction["user_id"] for interaction in interactions})
    items = sorted({interaction["parent_asin"] for interaction in interactions})
    history: dict[str, set[str]] = defaultdict(set)
    for interaction in interactions:
        history[interaction["user_id"]].add(interaction["parent_asin"])
    return (
        {user_id: index for index, user_id in enumerate(users)},
        {parent_asin: index for index, parent_asin in enumerate(items)},
        dict(history),
    )


def _build_edge_index(
    interactions: list[Interaction],
    user_to_index: Mapping[str, int],
    item_to_index: Mapping[str, int],
) -> torch.Tensor:
    num_users = len(user_to_index)
    user_nodes = [
        user_to_index[interaction["user_id"]] for interaction in interactions
    ]
    item_nodes = [
        num_users + item_to_index[interaction["parent_asin"]]
        for interaction in interactions
    ]
    return torch.tensor(
        [user_nodes + item_nodes, item_nodes + user_nodes],
        dtype=torch.long,
    )


class LightGCNRecommender:
    """Loadable LightGCN recommender that excludes known user interactions."""

    def __init__(
        self,
        model: LightGCN,
        config: LightGCNConfig,
        user_to_index: dict[str, int],
        item_to_index: dict[str, int],
        interacted_items_by_user: dict[str, set[str]],
        edge_index: torch.Tensor,
        *,
        device: str | torch.device,
    ) -> None:
        self.device = resolve_device(device)
        self.model = model.to(self.device)
        self.config = config
        self.user_to_index = user_to_index
        self.item_to_index = item_to_index
        self.index_to_item = [
            parent_asin
            for parent_asin, _ in sorted(
                item_to_index.items(), key=lambda entry: entry[1]
            )
        ]
        self.interacted_items_by_user = interacted_items_by_user
        self.edge_index = edge_index.to(self.device)

    @property
    def num_users(self) -> int:
        return len(self.user_to_index)

    @property
    def num_items(self) -> int:
        return len(self.item_to_index)

    @classmethod
    def from_training_interactions(
        cls,
        training_interactions: Iterable[Interaction],
        *,
        config: LightGCNConfig = LightGCNConfig(),
        seed: int = 0,
        device: str | torch.device = "cuda",
    ) -> LightGCNRecommender:
        interactions = list(training_interactions)
        if not interactions:
            raise ValueError("training interactions must not be empty")
        user_to_index, item_to_index, history = _build_mappings(interactions)
        edge_index = _build_edge_index(interactions, user_to_index, item_to_index)
        torch.manual_seed(seed)
        model = LightGCN(
            num_nodes=len(user_to_index) + len(item_to_index),
            embedding_dim=config.embedding_dim,
            num_layers=config.num_layers,
        )
        return cls(
            model,
            config,
            user_to_index,
            item_to_index,
            history,
            edge_index,
            device=device,
        )

    def save(self, path: Path) -> None:
        """Save everything needed for inference without Parquet inputs."""

        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(f"{path.suffix}.part")
        payload = {
            "format_version": MODEL_FORMAT_VERSION,
            "config": asdict(self.config),
            "state_dict": {
                key: value.detach().cpu()
                for key, value in self.model.state_dict().items()
            },
            "user_to_index": self.user_to_index,
            "item_to_index": self.item_to_index,
            "interacted_items_by_user": {
                user_id: sorted(items)
                for user_id, items in self.interacted_items_by_user.items()
            },
            "edge_index": self.edge_index.cpu(),
        }
        torch.save(payload, partial)
        partial.replace(path)

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        device: str | torch.device = "cuda",
    ) -> LightGCNRecommender:
        resolved_device = resolve_device(device)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["format_version"] != MODEL_FORMAT_VERSION:
            raise ValueError(
                f"unsupported LightGCN model format: {payload['format_version']}"
            )
        config = LightGCNConfig(**payload["config"])
        user_to_index = payload["user_to_index"]
        item_to_index = payload["item_to_index"]
        model = LightGCN(
            num_nodes=len(user_to_index) + len(item_to_index),
            embedding_dim=config.embedding_dim,
            num_layers=config.num_layers,
        )
        model.load_state_dict(payload["state_dict"])
        return cls(
            model,
            config,
            user_to_index,
            item_to_index,
            {
                user_id: set(items)
                for user_id, items in payload["interacted_items_by_user"].items()
            },
            payload["edge_index"],
            device=resolved_device,
        )

    def recommend(
        self,
        user_id: str,
        *,
        k: int = 10,
        excluded_items: Iterable[str] = (),
    ) -> list[str]:
        return self.recommend_batch(
            [user_id],
            k=k,
            excluded_items_by_user={user_id: excluded_items},
            batch_size=1,
        )[user_id]

    def recommend_batch(
        self,
        user_ids: Iterable[str],
        *,
        k: int = 10,
        excluded_items_by_user: Mapping[str, Iterable[str]] | None = None,
        batch_size: int = 256,
    ) -> dict[str, list[str]]:
        """Recommend in batches while computing propagated embeddings once."""

        if k < 0:
            raise ValueError("k must be non-negative")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        requested_user_ids = list(dict.fromkeys(user_ids))
        recommendations = {user_id: [] for user_id in requested_user_ids}
        known_user_ids = [
            user_id
            for user_id in requested_user_ids
            if user_id in self.user_to_index
        ]
        if k == 0 or not known_user_ids:
            return recommendations

        excluded_items_by_user = excluded_items_by_user or {}
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model.get_embedding(self.edge_index)
            item_embeddings = embeddings[self.num_users :]
            for start in range(0, len(known_user_ids), batch_size):
                batch_user_ids = known_user_ids[start : start + batch_size]
                user_indices = torch.tensor(
                    [self.user_to_index[user_id] for user_id in batch_user_ids],
                    dtype=torch.long,
                    device=self.device,
                )
                scores = embeddings[user_indices] @ item_embeddings.t()
                for row_index, user_id in enumerate(batch_user_ids):
                    excluded = set(self.interacted_items_by_user[user_id])
                    excluded.update(excluded_items_by_user.get(user_id, ()))
                    excluded_indices = [
                        self.item_to_index[parent_asin]
                        for parent_asin in excluded
                        if parent_asin in self.item_to_index
                    ]
                    if excluded_indices:
                        scores[row_index, excluded_indices] = -torch.inf

                ranked_indices = torch.argsort(
                    scores, dim=1, descending=True, stable=True
                )
                for row_index, user_id in enumerate(batch_user_ids):
                    ranked = ranked_indices[row_index]
                    finite = torch.isfinite(scores[row_index, ranked])
                    selected = ranked[finite][:k].tolist()
                    recommendations[user_id] = [
                        self.index_to_item[item_index] for item_index in selected
                    ]
        return recommendations


def _sample_negative_items(
    user_indices: torch.Tensor,
    interacted_item_indices_by_user: Mapping[int, set[int]],
    *,
    num_items: int,
    generator: torch.Generator,
) -> torch.Tensor:
    negatives: list[int] = []
    for user_index in user_indices.tolist():
        interacted_items = interacted_item_indices_by_user[user_index]
        if len(interacted_items) == num_items:
            raise ValueError(
                f"user index {user_index} has interacted with every training item"
            )
        candidate = int(
            torch.randint(num_items, (1,), generator=generator).item()
        )
        while candidate in interacted_items:
            candidate = int(
                torch.randint(num_items, (1,), generator=generator).item()
            )
        negatives.append(candidate)
    return torch.tensor(negatives, dtype=torch.long)


def train_lightgcn(
    training_interactions: Iterable[Interaction],
    *,
    config: LightGCNConfig = LightGCNConfig(),
    epochs: int = 5,
    learning_rate: float = 0.01,
    lambda_reg: float = 1e-4,
    seed: int = 0,
    device: str | torch.device = "cuda",
    epoch_callback: Callable[[int, float], None] | None = None,
) -> tuple[LightGCNRecommender, list[float]]:
    """Train LightGCN with full-graph propagation, BPR loss, and negatives."""

    if epochs <= 0:
        raise ValueError("epochs must be greater than zero")

    interactions = list(training_interactions)
    recommender = LightGCNRecommender.from_training_interactions(
        interactions,
        config=config,
        seed=seed,
        device=device,
    )
    user_indices = torch.tensor(
        [
            recommender.user_to_index[interaction["user_id"]]
            for interaction in interactions
        ],
        dtype=torch.long,
    )
    positive_item_indices = torch.tensor(
        [
            recommender.item_to_index[interaction["parent_asin"]]
            for interaction in interactions
        ],
        dtype=torch.long,
    )
    interacted_item_indices_by_user = {
        recommender.user_to_index[user_id]: {
            recommender.item_to_index[parent_asin] for parent_asin in items
        }
        for user_id, items in recommender.interacted_items_by_user.items()
    }
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(recommender.model.parameters(), lr=learning_rate)
    losses: list[float] = []

    for epoch in range(1, epochs + 1):
        negative_item_indices = _sample_negative_items(
            user_indices,
            interacted_item_indices_by_user,
            num_items=recommender.num_items,
            generator=generator,
        )
        user_nodes = user_indices.to(recommender.device)
        positive_nodes = (
            recommender.num_users + positive_item_indices
        ).to(recommender.device)
        negative_nodes = (
            recommender.num_users + negative_item_indices
        ).to(recommender.device)

        recommender.model.train()
        optimizer.zero_grad()
        embeddings = recommender.model.get_embedding(recommender.edge_index)
        positive_ranks = (embeddings[user_nodes] * embeddings[positive_nodes]).sum(
            dim=-1
        )
        negative_ranks = (embeddings[user_nodes] * embeddings[negative_nodes]).sum(
            dim=-1
        )
        node_id = torch.cat([user_nodes, positive_nodes, negative_nodes]).unique()
        loss = recommender.model.recommendation_loss(
            positive_ranks,
            negative_ranks,
            node_id=node_id,
            lambda_reg=lambda_reg,
        )
        loss.backward()
        optimizer.step()
        epoch_loss = loss.item()
        losses.append(epoch_loss)
        if epoch_callback is not None:
            epoch_callback(epoch, epoch_loss)
    return recommender, losses


def evaluate_lightgcn_recommender(
    recommender: LightGCNRecommender,
    target_interactions: Iterable[Interaction],
    *,
    k: int = 10,
    additional_history: Iterable[Interaction] = (),
    batch_size: int = 256,
) -> RankingMetrics:
    """Evaluate LightGCN in batches with the shared ranking metric formula."""

    targets_by_user = group_items_by_user(target_interactions)
    history_by_user = group_items_by_user(additional_history)
    recommendations_by_user = recommender.recommend_batch(
        targets_by_user,
        k=k,
        excluded_items_by_user=history_by_user,
        batch_size=batch_size,
    )
    return ranking_metrics_from_recommendations(
        targets_by_user,
        recommendations_by_user,
        k=k,
    )
