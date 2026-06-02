"""Train, save, reload, and evaluate the PyG LightGCN recommender."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch


from cartwise.retrieval.lightgcn import (
    LightGCNConfig,
    LightGCNRecommender,
    evaluate_lightgcn_recommender,
    resolve_device,
    train_lightgcn,
)
from cartwise.retrieval.popularity import (
    PopularityRecommender,
    evaluate_recommender,
    load_interactions,
)
from scripts.paths import METRICS_ROOT, MODELS_ROOT, PROCESSED_ROOTS
from scripts.pipeline.evaluate_popularity import format_metrics_row, write_metrics_csv

DEFAULT_SCOPE = "full"
PARAMETERS_COLUMN = "gcn_parameters"
LEGACY_PARAMETERS = "legacy: parameters not recorded"
LOSSES_COLUMN = "last_10_epoch_losses"
LEGACY_LOSSES = "legacy: losses not recorded"
EVALUATION_ONLY_LOSSES = "evaluation only: loaded saved model"
METRIC_NAMES = ("Recall", "NDCG", "HitRate")
DEFAULT_CONFIG = LightGCNConfig()
DEFAULT_EPOCHS = 5
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_LAMBDA_REG = 1e-4
DEFAULT_SEED = 0
DEFAULT_DEVICE = "cuda"
DEFAULT_EVAL_BATCH_SIZE = 256
DEFAULT_K = 10
SCOPE_PATHS = {
    "dev": (
        PROCESSED_ROOTS["dev"],
        MODELS_ROOT / "lightgcn" / "dev" / "lightgcn.pt",
        METRICS_ROOT / "dev" / "lightgcn.csv",
    ),
    "full": (
        PROCESSED_ROOTS["full"],
        MODELS_ROOT / "lightgcn" / "full" / "lightgcn.pt",
        METRICS_ROOT / "full" / "lightgcn.csv",
    ),
}
DEFAULT_PROCESSED_ROOT, DEFAULT_MODEL_OUTPUT, DEFAULT_METRICS_OUTPUT = SCOPE_PATHS[
    DEFAULT_SCOPE
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=SCOPE_PATHS, default=DEFAULT_SCOPE)
    parser.add_argument("--processed-root", type=Path)
    parser.add_argument("--model-output", type=Path)
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument(
        "--embedding-dim", type=int, default=DEFAULT_CONFIG.embedding_dim
    )
    parser.add_argument("--num-layers", type=int, default=DEFAULT_CONFIG.num_layers)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--lambda-reg", type=float, default=DEFAULT_LAMBDA_REG)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--eval-batch-size", type=int, default=DEFAULT_EVAL_BATCH_SIZE)
    parser.add_argument("--k", type=int, nargs="+", default=[DEFAULT_K])
    parser.add_argument("--evaluate-only", action="store_true")
    return parser.parse_args()


def resolve_paths(
    scope: str,
    processed_root: Path | None = None,
    model_output: Path | None = None,
    metrics_output: Path | None = None,
) -> tuple[Path, Path, Path]:
    defaults = SCOPE_PATHS[scope]
    return (
        processed_root or defaults[0],
        model_output or defaults[1],
        metrics_output or defaults[2],
    )


def serialize_parameters(
    *,
    config: LightGCNConfig,
    epochs: int,
    learning_rate: float,
    lambda_reg: float,
    seed: int,
    device: torch.device,
    eval_batch_size: int,
    k: int | list[int],
) -> str:
    """Record only non-default inputs needed to compare experiments."""

    ks = normalize_ks(k)
    serialized_k: int | list[int] = ks[0] if len(ks) == 1 else ks
    parameters = {
        "device": str(device),
        "embedding_dim": config.embedding_dim,
        "epochs": epochs,
        "eval_batch_size": eval_batch_size,
        "k": serialized_k,
        "lambda_reg": lambda_reg,
        "learning_rate": learning_rate,
        "num_layers": config.num_layers,
        "seed": seed,
    }
    defaults = {
        "device": DEFAULT_DEVICE,
        "embedding_dim": DEFAULT_CONFIG.embedding_dim,
        "epochs": DEFAULT_EPOCHS,
        "eval_batch_size": DEFAULT_EVAL_BATCH_SIZE,
        "k": DEFAULT_K,
        "lambda_reg": DEFAULT_LAMBDA_REG,
        "learning_rate": DEFAULT_LEARNING_RATE,
        "num_layers": DEFAULT_CONFIG.num_layers,
        "seed": DEFAULT_SEED,
    }
    return json.dumps(
        {key: value for key, value in parameters.items() if value != defaults[key]},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def serialize_last_losses(losses: list[float]) -> str:
    """Record the final losses without making the metrics CSV excessively wide."""

    return json.dumps(losses[-10:], separators=(",", ":"))


def normalize_ks(k: int | list[int]) -> list[int]:
    """Normalize one or more evaluation cutoffs while preserving their order."""

    ks = [k] if isinstance(k, int) else k
    if not ks or any(value <= 0 for value in ks):
        raise ValueError("--k values must be greater than zero")
    return list(dict.fromkeys(ks))


def metric_ks_from_fieldnames(fieldnames: list[str] | None) -> list[int]:
    """Return complete metric groups from an existing CSV header."""

    fields = set(fieldnames or [])
    metric_ks: list[int] = []
    for fieldname in fieldnames or []:
        if not fieldname.startswith("Recall@"):
            continue
        try:
            k = int(fieldname.removeprefix("Recall@"))
        except ValueError:
            continue
        if all(f"{metric}@{k}" in fields for metric in METRIC_NAMES):
            metric_ks.append(k)
    return list(dict.fromkeys(metric_ks))


def merge_metrics_rows(
    rows_by_k: list[list[dict[str, str | int]]],
) -> list[dict[str, str | int]]:
    """Combine metrics at multiple cutoffs into one row per model and split."""

    merged: dict[tuple[str, str], dict[str, str | int]] = {}
    for rows in rows_by_k:
        for row in rows:
            key = str(row["model"]), str(row["split"])
            merged.setdefault(key, {}).update(row)
    return list(merged.values())


def popularity_metrics_rows(
    train: list[dict[str, str]],
    valid: list[dict[str, str]],
    test: list[dict[str, str]],
    *,
    k: int,
) -> list[dict[str, str | int]]:
    recommender = PopularityRecommender(train)
    return [
        format_metrics_row(
            "popularity",
            "valid",
            evaluate_recommender(recommender, valid, k=k),
            k=k,
        ),
        format_metrics_row(
            "popularity",
            "test",
            evaluate_recommender(
                recommender,
                test,
                k=k,
                additional_history=valid,
            ),
            k=k,
        ),
    ]


def popularity_metrics_rows_for_ks(
    train: list[dict[str, str]],
    valid: list[dict[str, str]],
    test: list[dict[str, str]],
    *,
    ks: list[int],
) -> list[dict[str, str | int]]:
    return merge_metrics_rows(
        [popularity_metrics_rows(train, valid, test, k=k) for k in ks]
    )


def lightgcn_metrics_rows_for_ks(
    recommender: LightGCNRecommender,
    valid: list[dict[str, str]],
    test: list[dict[str, str]],
    *,
    ks: list[int],
    eval_batch_size: int,
) -> list[dict[str, str | int]]:
    return merge_metrics_rows(
        [
            [
                format_metrics_row(
                    "lightgcn",
                    "valid",
                    evaluate_lightgcn_recommender(
                        recommender,
                        valid,
                        k=k,
                        batch_size=eval_batch_size,
                    ),
                    k=k,
                ),
                format_metrics_row(
                    "lightgcn",
                    "test",
                    evaluate_lightgcn_recommender(
                        recommender,
                        test,
                        k=k,
                        additional_history=valid,
                        batch_size=eval_batch_size,
                    ),
                    k=k,
                ),
            ]
            for k in ks
        ]
    )


def append_experiment_metrics_csv(
    output: Path,
    popularity_rows: list[dict[str, str | int]],
    lightgcn_rows: list[dict[str, str | int]],
    *,
    parameters: str,
    last_losses: str,
    k: int | list[int],
) -> None:
    """Keep the baseline first and append each LightGCN experiment."""

    ks = normalize_ks(k)
    baseline_by_split: dict[str, dict[str, str | int]] = {}
    historical_rows: list[dict[str, str | int]] = []
    historical_ks: list[int] = []
    if output.exists():
        with output.open(newline="", encoding="utf-8") as input_file:
            reader = csv.DictReader(input_file)
            required_fields = {"model", "split", "users"}
            if not required_fields.issubset(reader.fieldnames or []):
                raise ValueError(
                    f"existing metrics CSV has an incompatible schema: {output}"
                )
            historical_ks = metric_ks_from_fieldnames(reader.fieldnames)
            for row in reader:
                if row["model"] == "popularity":
                    baseline_by_split[row["split"]] = row
                    continue
                row[PARAMETERS_COLUMN] = row.get(PARAMETERS_COLUMN) or LEGACY_PARAMETERS
                row[LOSSES_COLUMN] = row.get(LOSSES_COLUMN) or LEGACY_LOSSES
                historical_rows.append(row)

    for row in popularity_rows:
        baseline_by_split.setdefault(str(row["split"]), {}).update(row)
    baseline_rows = [
        {**row, PARAMETERS_COLUMN: "", LOSSES_COLUMN: ""}
        for row in baseline_by_split.values()
    ]
    new_rows = [
        {**row, PARAMETERS_COLUMN: parameters, LOSSES_COLUMN: last_losses}
        for row in lightgcn_rows
    ]
    write_metrics_csv(
        output,
        baseline_rows + historical_rows + new_rows,
        k=ks[-1],
        additional_ks=[*historical_ks, *ks[:-1]],
        extra_fieldnames=[PARAMETERS_COLUMN, LOSSES_COLUMN],
    )


def train_and_evaluate(
    processed_root: Path = DEFAULT_PROCESSED_ROOT,
    model_output: Path = DEFAULT_MODEL_OUTPUT,
    metrics_output: Path = DEFAULT_METRICS_OUTPUT,
    *,
    config: LightGCNConfig = DEFAULT_CONFIG,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    lambda_reg: float = DEFAULT_LAMBDA_REG,
    seed: int = DEFAULT_SEED,
    device: str = DEFAULT_DEVICE,
    eval_batch_size: int = DEFAULT_EVAL_BATCH_SIZE,
    k: int | list[int] = DEFAULT_K,
) -> list[dict[str, str | int]]:
    resolved_device = resolve_device(device)
    ks = normalize_ks(k)
    if resolved_device.type == "cuda":
        print(
            f"Using GPU: {resolved_device} ({torch.cuda.get_device_name(resolved_device)})"
        )
    else:
        print(f"Using device: {resolved_device}")

    train = load_interactions(processed_root / "interactions_train.parquet")
    valid = load_interactions(processed_root / "interactions_valid.parquet")
    test = load_interactions(processed_root / "interactions_test.parquet")
    print(f"Training LightGCN: {len(train):,} interactions, {epochs} epoch(s)")
    recommender, losses = train_lightgcn(
        train,
        config=config,
        epochs=epochs,
        learning_rate=learning_rate,
        lambda_reg=lambda_reg,
        seed=seed,
        device=resolved_device,
        epoch_callback=lambda epoch, loss: print(
            f"Epoch {epoch}: loss={loss:.6f}",
            flush=True,
        ),
    )

    recommender.save(model_output)
    print(f"Wrote model: {model_output}")
    loaded_recommender = LightGCNRecommender.load(
        model_output,
        device=resolved_device,
    )
    lightgcn_rows = lightgcn_metrics_rows_for_ks(
        loaded_recommender,
        valid,
        test,
        ks=ks,
        eval_batch_size=eval_batch_size,
    )
    parameters = serialize_parameters(
        config=config,
        epochs=epochs,
        learning_rate=learning_rate,
        lambda_reg=lambda_reg,
        seed=seed,
        device=resolved_device,
        eval_batch_size=eval_batch_size,
        k=ks,
    )
    append_experiment_metrics_csv(
        metrics_output,
        popularity_metrics_rows_for_ks(train, valid, test, ks=ks),
        lightgcn_rows,
        parameters=parameters,
        last_losses=serialize_last_losses(losses),
        k=ks,
    )
    print(f"Appended metrics: {metrics_output}")
    for row in lightgcn_rows:
        print(row)
    return lightgcn_rows


def evaluate_saved_model(
    processed_root: Path = DEFAULT_PROCESSED_ROOT,
    model_output: Path = DEFAULT_MODEL_OUTPUT,
    metrics_output: Path = DEFAULT_METRICS_OUTPUT,
    *,
    device: str = DEFAULT_DEVICE,
    eval_batch_size: int = DEFAULT_EVAL_BATCH_SIZE,
    k: int | list[int] = DEFAULT_K,
) -> list[dict[str, str | int]]:
    """Evaluate an existing checkpoint without retraining it."""

    resolved_device = resolve_device(device)
    ks = normalize_ks(k)
    train = load_interactions(processed_root / "interactions_train.parquet")
    valid = load_interactions(processed_root / "interactions_valid.parquet")
    test = load_interactions(processed_root / "interactions_test.parquet")
    recommender = LightGCNRecommender.load(model_output, device=resolved_device)
    print(f"Loaded model: {model_output}")
    lightgcn_rows = lightgcn_metrics_rows_for_ks(
        recommender,
        valid,
        test,
        ks=ks,
        eval_batch_size=eval_batch_size,
    )
    parameters = json.dumps(
        {
            "evaluation_only": True,
            "model_output": str(model_output),
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    append_experiment_metrics_csv(
        metrics_output,
        popularity_metrics_rows_for_ks(train, valid, test, ks=ks),
        lightgcn_rows,
        parameters=parameters,
        last_losses=EVALUATION_ONLY_LOSSES,
        k=ks,
    )
    print(f"Appended metrics: {metrics_output}")
    for row in lightgcn_rows:
        print(row)
    return lightgcn_rows


def main() -> None:
    args = parse_args()
    processed_root, model_output, metrics_output = resolve_paths(
        args.scope,
        processed_root=args.processed_root,
        model_output=args.model_output,
        metrics_output=args.metrics_output,
    )
    if args.evaluate_only:
        evaluate_saved_model(
            processed_root,
            model_output,
            metrics_output,
            device=args.device,
            eval_batch_size=args.eval_batch_size,
            k=args.k,
        )
    else:
        train_and_evaluate(
            processed_root,
            model_output,
            metrics_output,
            config=LightGCNConfig(
                embedding_dim=args.embedding_dim,
                num_layers=args.num_layers,
            ),
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            lambda_reg=args.lambda_reg,
            seed=args.seed,
            device=args.device,
            eval_batch_size=args.eval_batch_size,
            k=args.k,
        )


if __name__ == "__main__":
    main()
