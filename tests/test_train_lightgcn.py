from __future__ import annotations

import csv
import json
from pathlib import Path

from cartwise.retrieval.lightgcn import LightGCNConfig
from scripts.pipeline.train_lightgcn import (
    LEGACY_LOSSES,
    LEGACY_PARAMETERS,
    LOSSES_COLUMN,
    PARAMETERS_COLUMN,
    append_experiment_metrics_csv,
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
