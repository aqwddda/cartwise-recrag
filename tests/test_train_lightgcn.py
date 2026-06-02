from __future__ import annotations

import csv
import json
from pathlib import Path

from cartwise.retrieval.lightgcn import LightGCNConfig
from scripts.pipeline.train_lightgcn import (
    EVALUATION_ONLY_LOSSES,
    LEGACY_LOSSES,
    LEGACY_PARAMETERS,
    LOSSES_COLUMN,
    PARAMETERS_COLUMN,
    append_experiment_metrics_csv,
    evaluate_saved_model,
    merge_metrics_rows,
    serialize_last_losses,
    serialize_parameters,
)


def metrics_row(model: str, split: str, value: str) -> dict[str, str]:
    return {
        "model": model,
        "split": split,
        "users": "1",
        "Recall@10": value,
        "NDCG@10": value,
        "HitRate@10": value,
    }


def metrics_row_at_k(model: str, split: str, value: str, *, k: int) -> dict[str, str]:
    return {
        "model": model,
        "split": split,
        "users": "1",
        f"Recall@{k}": value,
        f"NDCG@{k}": value,
        f"HitRate@{k}": value,
    }


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as input_file:
        return list(csv.DictReader(input_file))


def test_append_experiment_metrics_keeps_baseline_first_and_preserves_history(
    tmp_path: Path,
) -> None:
    output = tmp_path / "lightgcn.csv"
    output.write_text(
        "model,split,users,Recall@10,NDCG@10,HitRate@10\n"
        "lightgcn,valid,1,0.1,0.1,0.1\n"
        "lightgcn,test,1,0.2,0.2,0.2\n",
        encoding="utf-8",
    )

    append_experiment_metrics_csv(
        output,
        [
            metrics_row("popularity", "valid", "0.3"),
            metrics_row("popularity", "test", "0.4"),
        ],
        [
            metrics_row("lightgcn", "valid", "0.5"),
            metrics_row("lightgcn", "test", "0.6"),
        ],
        parameters='{"epochs":5}',
        last_losses="[0.5,0.4]",
        k=10,
    )

    rows = read_rows(output)
    assert [row["model"] for row in rows] == [
        "popularity",
        "popularity",
        "lightgcn",
        "lightgcn",
        "lightgcn",
        "lightgcn",
    ]
    assert [row[PARAMETERS_COLUMN] for row in rows] == [
        "",
        "",
        LEGACY_PARAMETERS,
        LEGACY_PARAMETERS,
        '{"epochs":5}',
        '{"epochs":5}',
    ]
    assert [row[LOSSES_COLUMN] for row in rows] == [
        "",
        "",
        LEGACY_LOSSES,
        LEGACY_LOSSES,
        "[0.5,0.4]",
        "[0.5,0.4]",
    ]


def test_append_experiment_metrics_appends_to_existing_parameterized_rows(
    tmp_path: Path,
) -> None:
    output = tmp_path / "lightgcn.csv"
    baseline = [
        metrics_row("popularity", "valid", "0.1"),
        metrics_row("popularity", "test", "0.2"),
    ]

    append_experiment_metrics_csv(
        output,
        baseline,
        [metrics_row("lightgcn", "valid", "0.3")],
        parameters='{"epochs":5}',
        last_losses="[0.3]",
        k=10,
    )
    append_experiment_metrics_csv(
        output,
        baseline,
        [metrics_row("lightgcn", "valid", "0.4")],
        parameters='{"epochs":10}',
        last_losses="[0.2]",
        k=10,
    )

    rows = read_rows(output)
    assert [row[PARAMETERS_COLUMN] for row in rows] == [
        "",
        "",
        '{"epochs":5}',
        '{"epochs":10}',
    ]
    assert [row[LOSSES_COLUMN] for row in rows] == [
        "",
        "",
        "[0.3]",
        "[0.2]",
    ]


def test_append_experiment_metrics_adds_columns_for_a_new_k(
    tmp_path: Path,
) -> None:
    output = tmp_path / "lightgcn.csv"
    output.write_text(
        "model,split,users,Recall@10,NDCG@10,HitRate@10,"
        "gcn_parameters,last_10_epoch_losses\n"
        "popularity,valid,1,0.1,0.1,0.1,,\n"
        "popularity,test,1,0.2,0.2,0.2,,\n"
        "lightgcn,valid,1,0.3,0.3,0.3,{\"\"epochs\"\":5},[0.3]\n",
        encoding="utf-8",
    )

    append_experiment_metrics_csv(
        output,
        [
            metrics_row_at_k("popularity", "valid", "0.4", k=50),
            metrics_row_at_k("popularity", "test", "0.5", k=50),
        ],
        [metrics_row_at_k("lightgcn", "valid", "0.6", k=50)],
        parameters='{"epochs":10,"k":50}',
        last_losses="[0.2]",
        k=50,
    )

    rows = read_rows(output)
    assert list(rows[0]) == [
        "model",
        "split",
        "users",
        "Recall@10",
        "NDCG@10",
        "HitRate@10",
        "Recall@50",
        "NDCG@50",
        "HitRate@50",
        PARAMETERS_COLUMN,
        LOSSES_COLUMN,
    ]
    assert rows[0]["Recall@10"] == "0.1"
    assert rows[0]["Recall@50"] == "0.4"
    assert rows[1]["Recall@10"] == "0.2"
    assert rows[1]["Recall@50"] == "0.5"
    assert rows[2]["Recall@10"] == "0.3"
    assert rows[2]["Recall@50"] == ""
    assert rows[3]["Recall@10"] == ""
    assert rows[3]["Recall@50"] == "0.6"


def test_merge_metrics_rows_combines_multiple_ks_for_the_same_split() -> None:
    rows = merge_metrics_rows(
        [
            [
                metrics_row_at_k("lightgcn", "valid", "0.1", k=10),
                metrics_row_at_k("lightgcn", "test", "0.2", k=10),
            ],
            [
                metrics_row_at_k("lightgcn", "valid", "0.3", k=50),
                metrics_row_at_k("lightgcn", "test", "0.4", k=50),
            ],
        ]
    )

    assert rows == [
        {
            "model": "lightgcn",
            "split": "valid",
            "users": "1",
            "Recall@10": "0.1",
            "NDCG@10": "0.1",
            "HitRate@10": "0.1",
            "Recall@50": "0.3",
            "NDCG@50": "0.3",
            "HitRate@50": "0.3",
        },
        {
            "model": "lightgcn",
            "split": "test",
            "users": "1",
            "Recall@10": "0.2",
            "NDCG@10": "0.2",
            "HitRate@10": "0.2",
            "Recall@50": "0.4",
            "NDCG@50": "0.4",
            "HitRate@50": "0.4",
        },
    ]


def test_evaluate_saved_model_loads_checkpoint_without_training(
    tmp_path: Path,
    monkeypatch,
) -> None:
    processed_root = tmp_path / "processed"
    model_output = tmp_path / "lightgcn.pt"
    metrics_output = tmp_path / "lightgcn.csv"
    interactions = {
        "interactions_train.parquet": [{"user_id": "U1", "parent_asin": "P1"}],
        "interactions_valid.parquet": [{"user_id": "U1", "parent_asin": "P2"}],
        "interactions_test.parquet": [{"user_id": "U1", "parent_asin": "P3"}],
    }
    loaded_recommender = object()
    calls: list[tuple[object, int]] = []

    monkeypatch.setattr(
        "scripts.pipeline.train_lightgcn.load_interactions",
        lambda path: interactions[path.name],
    )
    monkeypatch.setattr(
        "scripts.pipeline.train_lightgcn.LightGCNRecommender.load",
        lambda path, *, device: loaded_recommender,
    )
    monkeypatch.setattr(
        "scripts.pipeline.train_lightgcn.evaluate_lightgcn_recommender",
        lambda recommender, target, *, k, additional_history=(), batch_size: (
            calls.append((recommender, k))
            or type(
                "Metrics",
                (),
                {"users": 1, "recall": k / 100, "ndcg": k / 100, "hit_rate": 1.0},
            )()
        ),
    )
    monkeypatch.setattr(
        "scripts.pipeline.train_lightgcn.popularity_metrics_rows",
        lambda train, valid, test, *, k: [
            metrics_row_at_k("popularity", "valid", str(k), k=k),
            metrics_row_at_k("popularity", "test", str(k), k=k),
        ],
    )

    rows = evaluate_saved_model(
        processed_root,
        model_output,
        metrics_output,
        device="cpu",
        k=[10, 50],
    )

    assert calls == [
        (loaded_recommender, 10),
        (loaded_recommender, 10),
        (loaded_recommender, 50),
        (loaded_recommender, 50),
    ]
    assert rows[0]["Recall@10"] == "0.100000"
    assert rows[0]["Recall@50"] == "0.500000"
    written_rows = read_rows(metrics_output)
    assert written_rows[2][LOSSES_COLUMN] == EVALUATION_ONLY_LOSSES


def test_serialize_parameters_records_comparable_training_inputs() -> None:
    parameters = serialize_parameters(
        config=LightGCNConfig(embedding_dim=32, num_layers=2),
        epochs=20,
        learning_rate=0.005,
        lambda_reg=1e-4,
        seed=7,
        device="cuda",
        eval_batch_size=128,
        k=10,
    )

    assert json.loads(parameters) == {
        "embedding_dim": 32,
        "epochs": 20,
        "eval_batch_size": 128,
        "learning_rate": 0.005,
        "num_layers": 2,
        "seed": 7,
    }


def test_serialize_parameters_omits_all_defaults() -> None:
    parameters = serialize_parameters(
        config=LightGCNConfig(),
        epochs=5,
        learning_rate=0.01,
        lambda_reg=1e-4,
        seed=0,
        device="cuda",
        eval_batch_size=256,
        k=10,
    )

    assert parameters == "{}"


def test_serialize_last_losses_keeps_at_most_final_ten_epochs() -> None:
    losses = [float(loss) for loss in range(12)]

    assert json.loads(serialize_last_losses(losses)) == losses[-10:]
