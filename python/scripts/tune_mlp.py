from __future__ import annotations

import argparse
import shutil
from datetime import UTC, datetime
from typing import Any

import numpy as np
import torch

from deeponet_irrigation.dataset import (
    PreparedTemporalDataset,
    fit_residual_normalization,
    make_data_loader,
)
from deeponet_irrigation.evaluation import collect_predictions
from deeponet_irrigation.metrics import metrics_by_horizon
from deeponet_irrigation.models import MLPBaseline
from deeponet_irrigation.project_paths import REPOSITORY_ROOT
from deeponet_irrigation.training import (
    TrainingConfig,
    select_best_experiment,
    select_device,
    set_deterministic_seed,
    train_mlp,
    write_json,
)


ROOT = REPOSITORY_ROOT
DATA_DIR = ROOT / "data" / "processed" / "arnesano_v2"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
EXPERIMENT_LOG = RESULTS_DIR / "mlp_experiments.json"
SELECTED_CHECKPOINT = MODELS_DIR / "mlp_baseline_best.pt"
FINAL_TEST_RESULTS = RESULTS_DIR / "final_baseline_metrics.json"


EXPERIMENT_CONFIGS = (
    {"hidden_sizes": (128, 64), "learning_rate": 1e-3, "loss": "mse", "weight_decay": 0.0},
    {"hidden_sizes": (128, 64), "learning_rate": 1e-3, "loss": "huber", "weight_decay": 0.0},
    {"hidden_sizes": (256, 128), "learning_rate": 1e-3, "loss": "mse", "weight_decay": 0.0},
    {"hidden_sizes": (64, 32), "learning_rate": 1e-3, "loss": "mse", "weight_decay": 0.0},
    {"hidden_sizes": (128, 64), "learning_rate": 3e-4, "loss": "mse", "weight_decay": 0.0},
    {"hidden_sizes": (128, 64), "learning_rate": 1e-3, "loss": "mse", "weight_decay": 1e-5},
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a bounded validation-only residual MLP search")
    parser.add_argument("--max-experiments", type=int, default=6, choices=range(1, 13))
    return parser.parse_args()


def success_criteria(metrics: dict[str, Any]) -> bool:
    overall = metrics["overall"]
    horizons = metrics["by_horizon"]
    return bool(
        overall["rmse"] < 1.2174450185045171
        and (
            horizons["12h"]["rmse"] < 1.2048613286416154
            or horizons["24h"]["rmse"] < 1.4692617733849058
        )
        and overall["mae"] < 1.0
    )


def baseline_metrics() -> tuple[dict[str, Any], dict[str, Any]]:
    path = RESULTS_DIR / "validation_predictions.npz"
    if not path.exists():
        raise FileNotFoundError(
            "Missing prior validation_predictions.npz needed for the frozen direct baseline"
        )
    with np.load(path) as values:
        target = values["target"]
        horizon = values["horizon_hours"]
        persistence = metrics_by_horizon(target, values["persistence"], horizon)
        direct = metrics_by_horizon(target, values["mlp"], horizon)
    return persistence, direct


def print_table(
    persistence: dict[str, Any],
    direct: dict[str, Any],
    experiments: list[dict[str, object]],
) -> None:
    rows = [
        ("persistence", persistence["overall"]),
        ("direct-existing", direct["overall"]),
    ]
    rows.extend(
        (str(item["experiment_id"]), item["validation_metrics"]["overall"])
        for item in experiments
    )
    rows.sort(key=lambda item: item[1]["rmse"])
    print("\nValidation comparison (sorted by overall RMSE)")
    print(f"{'Model':<20} {'MAE':>10} {'RMSE':>10} {'R2':>10}")
    for label, metrics in rows:
        print(
            f"{label:<20} {metrics['mae']:>10.4f} "
            f"{metrics['rmse']:>10.4f} {metrics['r2']:>10.4f}"
        )


def main() -> None:
    args = parse_args()
    if FINAL_TEST_RESULTS.exists():
        raise SystemExit(
            "Final test results already exist; validation tuning is permanently closed."
        )
    train_dataset = PreparedTemporalDataset(DATA_DIR, "train")
    validation_dataset = PreparedTemporalDataset(DATA_DIR, "validation")
    residual_normalization = fit_residual_normalization(train_dataset)
    persistence, direct = baseline_metrics()
    device = select_device()
    experiments: list[dict[str, object]] = []
    print(f"Device: {device}")
    print(f"Residual normalization (train only): {residual_normalization}")

    for number, search_config in enumerate(
        EXPERIMENT_CONFIGS[: args.max_experiments], start=1
    ):
        experiment_id = f"residual_{number:02d}"
        config = TrainingConfig(
            batch_size=512,
            learning_rate=search_config["learning_rate"],
            max_epochs=50,
            patience=8,
            seed=42,
            loss=search_config["loss"],
            weight_decay=search_config["weight_decay"],
        )
        train_loader = make_data_loader(
            train_dataset, batch_size=config.batch_size, shuffle=True, seed=config.seed
        )
        validation_loader = make_data_loader(
            validation_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            seed=config.seed,
        )
        set_deterministic_seed(config.seed)
        model = MLPBaseline(
            history_steps=train_dataset.history_steps,
            feature_count=train_dataset.feature_count,
            hidden_sizes=search_config["hidden_sizes"],
            target_mean=float(residual_normalization["mean"]),
            target_std=float(residual_normalization["std"]),
            target_mode="residual",
            soil_feature_index=train_dataset.soil_index,
            soil_mean=train_dataset.soil_mean,
            soil_std=train_dataset.soil_std,
        )
        checkpoint_path = MODELS_DIR / "experiments" / f"{experiment_id}.pt"
        print(f"\n{experiment_id}: {search_config}")
        training_result = train_mlp(
            model,
            train_loader,
            validation_loader,
            device=device,
            config=config,
            checkpoint_path=checkpoint_path,
            preprocessing_metadata=train_dataset.metadata,
        )
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        predictions = collect_predictions(
            validation_loader,
            validation_dataset.metadata,
            device=device,
            model=model,
        )
        validation_metrics = metrics_by_horizon(
            predictions["target"], predictions["mlp"], predictions["horizon_hours"]
        )
        experiment: dict[str, object] = {
            "experiment_id": experiment_id,
            "target_mode": "residual",
            "architecture": list(search_config["hidden_sizes"]),
            "learning_rate": config.learning_rate,
            "loss": config.loss,
            "weight_decay": config.weight_decay,
            "batch_size": config.batch_size,
            "seed": config.seed,
            "best_epoch": training_result["best_epoch"],
            "best_validation_loss": training_result["best_validation_loss"],
            "validation_metrics": validation_metrics,
            "checkpoint": str(checkpoint_path),
            "duration_seconds": training_result["duration_seconds"],
            "device": training_result["device"],
            "gpu_name": training_result["gpu_name"],
        }
        experiments.append(experiment)
        interim = {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "selection_uses": "validation metrics only",
            "residual_normalization": residual_normalization,
            "persistence_validation_metrics": persistence,
            "direct_validation_metrics": direct,
            "experiments": experiments,
        }
        write_json(EXPERIMENT_LOG, interim)
        print_table(persistence, direct, experiments)
        if success_criteria(validation_metrics):
            print("Success criteria met; stopping the bounded search early.")
            break

    best = select_best_experiment(experiments)
    SELECTED_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(best["checkpoint"]), SELECTED_CHECKPOINT)
    best_model_checkpoint = torch.load(
        SELECTED_CHECKPOINT, map_location=device, weights_only=True
    )
    selected_model = MLPBaseline(
        history_steps=train_dataset.history_steps,
        feature_count=train_dataset.feature_count,
        hidden_sizes=tuple(best["architecture"]),
        target_mean=float(residual_normalization["mean"]),
        target_std=float(residual_normalization["std"]),
        target_mode="residual",
        soil_feature_index=train_dataset.soil_index,
        soil_mean=train_dataset.soil_mean,
        soil_std=train_dataset.soil_std,
    ).to(device)
    selected_model.load_state_dict(best_model_checkpoint["model_state_dict"])
    selected_predictions = collect_predictions(
        make_data_loader(validation_dataset, batch_size=1024, shuffle=False),
        validation_dataset.metadata,
        device=device,
        model=selected_model,
    )
    np.savez_compressed(
        RESULTS_DIR / "selected_validation_predictions.npz", **selected_predictions
    )

    final_log = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "selection_uses": "validation metrics only; test metrics are not loaded",
        "selection_priority": [
            "overall_rmse",
            "24h_rmse",
            "12h_rmse",
            "overall_r2",
            "overall_mae",
        ],
        "residual_normalization": residual_normalization,
        "persistence_validation_metrics": persistence,
        "direct_validation_metrics": direct,
        "experiments": experiments,
        "selected_experiment_id": best["experiment_id"],
        "selected_checkpoint": str(SELECTED_CHECKPOINT),
        "success_criteria_met": success_criteria(best["validation_metrics"]),
    }
    write_json(EXPERIMENT_LOG, final_log)
    print_table(persistence, direct, experiments)
    print(f"Selected {best['experiment_id']} -> {SELECTED_CHECKPOINT}")
    print(f"Experiment log: {EXPERIMENT_LOG}")


if __name__ == "__main__":
    main()
