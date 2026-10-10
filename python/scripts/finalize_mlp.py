from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from typing import Any

import numpy as np
import torch

from deeponet_irrigation.dataset import (
    PreparedTemporalDataset,
    fit_residual_normalization,
    fit_temporal_feature_normalization,
    make_data_loader,
)
from deeponet_irrigation.evaluation import collect_predictions, load_mlp_checkpoint
from deeponet_irrigation.metrics import (
    error_by_change_status,
    metrics_by_horizon,
    prediction_bias,
)
from deeponet_irrigation.models import MLPBaseline
from deeponet_irrigation.project_paths import REPOSITORY_ROOT
from deeponet_irrigation.temporal_features import TREND_FEATURES, WATER_FEATURES
from deeponet_irrigation.training import (
    TrainingConfig,
    select_device,
    set_deterministic_seed,
    train_mlp,
    update_checkpoint_metadata,
    write_json,
)


ROOT = REPOSITORY_ROOT
DATA_DIR = ROOT / "data" / "processed" / "arnesano_v2"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
LOG_PATH = RESULTS_DIR / "mlp_final_feature_experiments.json"
CONTROL_CHECKPOINT = MODELS_DIR / "mlp_baseline_control.pt"
CURRENT_CHECKPOINT = MODELS_DIR / "mlp_baseline_best.pt"
FINAL_CHECKPOINT = MODELS_DIR / "mlp_baseline_final.pt"
FINAL_TEST_RESULTS = RESULTS_DIR / "final_frozen_mlp_metrics.json"
PREDICTIONS_DIR = RESULTS_DIR / "final_feature_validation_predictions"
MAX_RUNS = 6

EXPERIMENTS: dict[str, dict[str, Any]] = {
    "trends_s42": {"features": TREND_FEATURES, "seed": 42},
    "water_s42": {"features": WATER_FEATURES, "seed": 42},
    "combined_s42": {"features": TREND_FEATURES + WATER_FEATURES, "seed": 42},
    "trends_s123": {"features": TREND_FEATURES, "seed": 123},
    "trends_s456": {"features": TREND_FEATURES, "seed": 456},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final bounded temporal-feature MLP pass")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--run", choices=tuple(EXPERIMENTS))
    action.add_argument("--select", metavar="EXPERIMENT_ID_OR_CONTROL")
    action.add_argument("--summary", action="store_true")
    return parser.parse_args()


def summarize_predictions(predictions: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        "validation_metrics": metrics_by_horizon(
            predictions["target"],
            predictions["mlp"],
            predictions["horizon_hours"],
        ),
        "validation_change_status": error_by_change_status(
            predictions["target"],
            predictions["mlp"],
            predictions["persistence"],
            tolerance=0.5,
        ),
        "validation_bias": prediction_bias(
            predictions["target"], predictions["mlp"]
        ),
    }


def load_control() -> dict[str, Any]:
    improvement = json.loads(
        (RESULTS_DIR / "mlp_improvement_experiments.json").read_text(
            encoding="utf-8"
        )
    )
    selected = improvement["selected"]
    return {
        "id": "control",
        "checkpoint": str(CONTROL_CHECKPOINT),
        "features": [],
        "validation_metrics": selected["validation_metrics"],
        "validation_change_status": selected["validation_change_status"],
        "validation_bias": selected["validation_bias"],
        "seed_robustness": improvement["seed_robustness"]["256_128_64_mse"],
    }


def load_log() -> dict[str, Any]:
    if LOG_PATH.exists():
        return json.loads(LOG_PATH.read_text(encoding="utf-8"))
    if not CURRENT_CHECKPOINT.exists():
        raise FileNotFoundError("Missing current best MLP checkpoint")
    CONTROL_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CURRENT_CHECKPOINT, CONTROL_CHECKPOINT)
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "selection_uses": "train and validation only; test artifacts are not loaded",
        "maximum_training_runs": MAX_RUNS,
        "control": load_control(),
        "experiments": [],
    }


def raw_input_scaling(metadata: dict[str, Any]) -> tuple[tuple[str, ...], tuple[float, ...], tuple[float, ...]]:
    order = tuple(metadata["feature_order"])
    statistics = metadata["normalization"]["statistics"]
    means = tuple(float(statistics[name]["mean"]) for name in order)
    stds = tuple(float(statistics[name]["std"]) for name in order)
    return order, means, stds


def run_experiment(experiment_id: str) -> None:
    if FINAL_TEST_RESULTS.exists():
        raise SystemExit("The final MLP test exists; MLP training is permanently closed.")
    log = load_log()
    if len(log["experiments"]) >= MAX_RUNS:
        raise SystemExit("The six-run final-pass limit has been reached.")
    if any(item["id"] == experiment_id for item in log["experiments"]):
        raise SystemExit(f"Experiment {experiment_id} already exists")

    specification = EXPERIMENTS[experiment_id]
    train_dataset = PreparedTemporalDataset(DATA_DIR, "train")
    validation_dataset = PreparedTemporalDataset(DATA_DIR, "validation")
    residual = fit_residual_normalization(train_dataset)
    temporal = fit_temporal_feature_normalization(
        train_dataset, specification["features"]
    )
    order, means, stds = raw_input_scaling(train_dataset.metadata)
    config = TrainingConfig(
        batch_size=512,
        learning_rate=1e-3,
        max_epochs=50,
        patience=8,
        seed=specification["seed"],
        loss="mse",
        weight_decay=0.0,
        change_weighting=None,
    )
    train_loader = make_data_loader(
        train_dataset, batch_size=config.batch_size, shuffle=True, seed=config.seed
    )
    validation_loader = make_data_loader(
        validation_dataset, batch_size=config.batch_size, shuffle=False, seed=config.seed
    )
    set_deterministic_seed(config.seed)
    model = MLPBaseline(
        history_steps=train_dataset.history_steps,
        feature_count=train_dataset.feature_count,
        hidden_sizes=(256, 128, 64),
        target_mean=0.0,
        target_std=float(residual["std"]),
        target_mode="residual",
        soil_feature_index=train_dataset.soil_index,
        soil_mean=train_dataset.soil_mean,
        soil_std=train_dataset.soil_std,
        residual_scaling_mode="std_only",
        output_initialization="zero",
        temporal_feature_names=specification["features"],
        temporal_feature_means=tuple(temporal["means"]),
        temporal_feature_stds=tuple(temporal["stds"]),
        input_feature_order=order,
        input_feature_means=means,
        input_feature_stds=stds,
    )
    checkpoint_path = MODELS_DIR / "final_feature_experiments" / f"{experiment_id}.pt"
    device = select_device()
    print(f"Running {experiment_id} on {device}: {specification['features']}")
    training = train_mlp(
        model,
        train_loader,
        validation_loader,
        device=device,
        config=config,
        checkpoint_path=checkpoint_path,
        preprocessing_metadata=train_dataset.metadata,
    )
    model, _ = load_mlp_checkpoint(checkpoint_path, train_dataset.metadata, device)
    predictions = collect_predictions(
        validation_loader, validation_dataset.metadata, device=device, model=model
    )
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    prediction_path = PREDICTIONS_DIR / f"{experiment_id}.npz"
    np.savez_compressed(prediction_path, **predictions)
    experiment = {
        "id": experiment_id,
        "features": list(specification["features"]),
        "feature_normalization": temporal,
        "input_dimension": 721 + len(specification["features"]),
        "architecture": [256, 128, 64],
        "seed": config.seed,
        "best_epoch": training["best_epoch"],
        "best_validation_loss": training["best_validation_loss"],
        **summarize_predictions(predictions),
        "checkpoint": str(checkpoint_path),
        "validation_predictions": str(prediction_path),
        "duration_seconds": training["duration_seconds"],
        "device": training["device"],
        "gpu_name": training["gpu_name"],
    }
    log["experiments"].append(experiment)
    log["updated_at_utc"] = datetime.now(UTC).isoformat()
    write_json(LOG_PATH, log)
    print_summary(log)


def select_final(identifier: str) -> None:
    if FINAL_TEST_RESULTS.exists():
        raise SystemExit("The final MLP test exists; selection is permanently closed.")
    log = load_log()
    selected = (
        log["control"]
        if identifier == "control"
        else next(item for item in log["experiments"] if item["id"] == identifier)
    )
    trend_runs = [
        item
        for item in log["experiments"]
        if item["id"] in {"trends_s42", "trends_s123", "trends_s456"}
    ]
    if len(trend_runs) == 3:
        rmse_values = [
            item["validation_metrics"]["overall"]["rmse"] for item in trend_runs
        ]
        log["trend_seed_robustness"] = {
            "seeds": [item["seed"] for item in trend_runs],
            "overall_rmse_values": rmse_values,
            "overall_rmse_mean": float(np.mean(rmse_values)),
            "overall_rmse_std": float(np.std(rmse_values)),
        }
    shutil.copy2(selected["checkpoint"], FINAL_CHECKPOINT)
    update_checkpoint_metadata(
        FINAL_CHECKPOINT,
        {
            "final_mlp": True,
            "final_selection": {
                "uses": "validation only",
                "id": identifier,
                "features": selected["features"],
                "validation_metrics": selected["validation_metrics"],
                "validation_change_status": selected["validation_change_status"],
                "validation_bias": selected["validation_bias"],
                "seed_robustness": log.get("trend_seed_robustness"),
            },
        },
    )
    log["selected"] = {
        "id": identifier,
        "checkpoint": str(FINAL_CHECKPOINT),
        "features": selected["features"],
        "validation_metrics": selected["validation_metrics"],
        "validation_change_status": selected["validation_change_status"],
        "validation_bias": selected["validation_bias"],
        "seed_robustness": log.get("trend_seed_robustness"),
        "selected_at_utc": datetime.now(UTC).isoformat(),
    }
    write_json(LOG_PATH, log)
    print(f"Final MLP frozen: {identifier} -> {FINAL_CHECKPOINT}")


def print_summary(log: dict[str, Any]) -> None:
    control_rmse = log["control"]["validation_metrics"]["overall"]["rmse"]
    print("\nFinal temporal-feature experiments")
    print(
        f"{'ID':<20} {'Features':>8} {'Seed':>5} {'MAE':>8} {'RMSE':>8} "
        f"{'Gain %':>8} {'12h':>8} {'24h':>8} {'Change':>8}"
    )
    for item in log["experiments"]:
        metrics = item["validation_metrics"]
        rmse = metrics["overall"]["rmse"]
        gain = 100.0 * (control_rmse - rmse) / control_rmse
        print(
            f"{item['id']:<20} {len(item['features']):>8d} {item['seed']:>5d} "
            f"{metrics['overall']['mae']:>8.4f} {rmse:>8.4f} {gain:>8.2f} "
            f"{metrics['by_horizon']['12h']['rmse']:>8.4f} "
            f"{metrics['by_horizon']['24h']['rmse']:>8.4f} "
            f"{item['validation_change_status']['changing']['rmse']:>8.4f}"
        )


def main() -> None:
    args = parse_args()
    if args.run:
        run_experiment(args.run)
    elif args.select:
        select_final(args.select)
    else:
        print_summary(load_log())


if __name__ == "__main__":
    main()
