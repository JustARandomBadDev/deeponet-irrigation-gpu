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
    make_data_loader,
    summarize_change_weighting,
)
from deeponet_irrigation.evaluation import (
    calibrate_residual_prediction,
    collect_predictions,
    load_mlp_checkpoint,
)
from deeponet_irrigation.metrics import (
    error_by_change_status,
    metrics_by_horizon,
    prediction_bias,
)
from deeponet_irrigation.models import MLPBaseline
from deeponet_irrigation.project_paths import REPOSITORY_ROOT
from deeponet_irrigation.training import (
    TrainingConfig,
    select_device,
    set_deterministic_seed,
    train_mlp,
    update_checkpoint_metadata,
    validation_selection_key,
    write_json,
)


ROOT = REPOSITORY_ROOT
DATA_DIR = ROOT / "data" / "processed" / "arnesano_v2"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
LOG_PATH = RESULTS_DIR / "mlp_improvement_experiments.json"
PREDICTIONS_DIR = RESULTS_DIR / "improvement_validation_predictions"
SELECTED_CHECKPOINT = MODELS_DIR / "mlp_baseline_best.pt"
NEW_FINAL_RESULTS = RESULTS_DIR / "final_improved_baseline_metrics.json"
MAX_NEW_RUNS = 12

MILD_CHANGE_WEIGHTING = {
    "change_threshold": 0.5,
    "large_change_threshold": 5.0,
    "change_weight": 1.5,
    "large_change_weight": 2.0,
}

# Deliberately small, sequential set. Additional seed/architecture entries are
# added only after earlier validation results justify them.
EXPERIMENTS: dict[str, dict[str, Any]] = {
    "pc_mse_s42": {
        "hidden_sizes": (128, 64),
        "loss": "mse",
        "change_weighting": None,
        "seed": 42,
    },
    "pc_huber_s42": {
        "hidden_sizes": (128, 64),
        "loss": "huber",
        "change_weighting": None,
        "seed": 42,
    },
    "pc_weighted_mse_s42": {
        "hidden_sizes": (128, 64),
        "loss": "mse",
        "change_weighting": MILD_CHANGE_WEIGHTING,
        "seed": 42,
    },
    "pc_weighted_huber_s42": {
        "hidden_sizes": (128, 64),
        "loss": "huber",
        "change_weighting": MILD_CHANGE_WEIGHTING,
        "seed": 42,
    },
    "pc_large_mse_s42": {
        "hidden_sizes": (256, 128, 64),
        "loss": "mse",
        "change_weighting": None,
        "seed": 42,
    },
    "pc_mse_s123": {
        "hidden_sizes": (128, 64),
        "loss": "mse",
        "change_weighting": None,
        "seed": 123,
    },
    "pc_mse_s456": {
        "hidden_sizes": (128, 64),
        "loss": "mse",
        "change_weighting": None,
        "seed": 456,
    },
    "pc_large_mse_s123": {
        "hidden_sizes": (256, 128, 64),
        "loss": "mse",
        "change_weighting": None,
        "seed": 123,
    },
    "pc_large_mse_s456": {
        "hidden_sizes": (256, 128, 64),
        "loss": "mse",
        "change_weighting": None,
        "seed": 456,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run and calibrate bounded persistence-centered MLP experiments"
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--run", choices=tuple(EXPERIMENTS))
    action.add_argument("--calibrate", metavar="EXPERIMENT_ID")
    action.add_argument("--select", metavar="EXPERIMENT_ID")
    action.add_argument("--summary", action="store_true")
    return parser.parse_args()


def load_log() -> dict[str, Any]:
    if LOG_PATH.exists():
        return json.loads(LOG_PATH.read_text(encoding="utf-8"))
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "selection_uses": "train and validation only; no test artifact is loaded",
        "maximum_new_training_runs": MAX_NEW_RUNS,
        "experiments": [],
        "calibration_results": {},
    }


def load_validation_baselines() -> dict[str, Any]:
    with np.load(RESULTS_DIR / "validation_predictions.npz") as direct:
        target = direct["target"].copy()
        horizon = direct["horizon_hours"].copy()
        current = direct["persistence"].copy()
        direct_prediction = direct["mlp"].copy()
    with np.load(RESULTS_DIR / "selected_validation_predictions.npz") as residual:
        if not np.array_equal(residual["target"], target):
            raise ValueError("Original residual targets use different validation data")
        if not np.array_equal(residual["horizon_hours"], horizon):
            raise ValueError("Original residual horizons use different validation data")
        if not np.array_equal(residual["persistence"], current):
            raise ValueError("Original residual origins use different validation data")
        residual_prediction = residual["mlp"].copy()

    def summarize(prediction: np.ndarray) -> dict[str, Any]:
        return {
            "metrics": metrics_by_horizon(target, prediction, horizon),
            "change_status": error_by_change_status(
                target, prediction, current, tolerance=0.5
            ),
            "bias": prediction_bias(target, prediction),
        }

    return {
        "persistence": summarize(current),
        "direct_mlp": summarize(direct_prediction),
        "original_residual_mlp": summarize(residual_prediction),
    }


def summarize_prediction(
    target: np.ndarray,
    prediction: np.ndarray,
    horizon: np.ndarray,
    current: np.ndarray,
) -> dict[str, Any]:
    return {
        "validation_metrics": metrics_by_horizon(target, prediction, horizon),
        "validation_change_status": error_by_change_status(
            target, prediction, current, tolerance=0.5
        ),
        "validation_bias": prediction_bias(target, prediction),
    }


def run_experiment(experiment_id: str) -> None:
    if NEW_FINAL_RESULTS.exists():
        raise SystemExit("Final improved test results exist; training is closed.")
    log = load_log()
    if len(log["experiments"]) >= MAX_NEW_RUNS:
        raise SystemExit("The 12-run limit has been reached.")
    if any(item["id"] == experiment_id for item in log["experiments"]):
        raise SystemExit(f"Experiment {experiment_id} already exists.")

    specification = EXPERIMENTS[experiment_id]
    train_dataset = PreparedTemporalDataset(DATA_DIR, "train")
    validation_dataset = PreparedTemporalDataset(DATA_DIR, "validation")
    residual_statistics = fit_residual_normalization(train_dataset)
    weighting = specification["change_weighting"]
    config = TrainingConfig(
        batch_size=512,
        learning_rate=1e-3,
        max_epochs=50,
        patience=8,
        seed=specification["seed"],
        loss=specification["loss"],
        weight_decay=0.0,
        change_weighting=weighting,
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
        hidden_sizes=specification["hidden_sizes"],
        target_mean=0.0,
        target_std=float(residual_statistics["std"]),
        target_mode="residual",
        soil_feature_index=train_dataset.soil_index,
        soil_mean=train_dataset.soil_mean,
        soil_std=train_dataset.soil_std,
        residual_scaling_mode="std_only",
        output_initialization="zero",
    )
    checkpoint_path = MODELS_DIR / "improvement_experiments" / f"{experiment_id}.pt"
    device = select_device()
    print(f"Running {experiment_id} on {device}: {specification}")
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
    summary = summarize_prediction(
        predictions["target"],
        predictions["mlp"],
        predictions["horizon_hours"],
        predictions["persistence"],
    )
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    prediction_path = PREDICTIONS_DIR / f"{experiment_id}.npz"
    np.savez_compressed(prediction_path, **predictions)

    experiment = {
        "id": experiment_id,
        "target_scaling_mode": "std_only",
        "residual_scale": float(residual_statistics["std"]),
        "architecture": list(specification["hidden_sizes"]),
        "output_initialization": "zero",
        "loss": specification["loss"],
        "loss_configuration": {
            "huber_delta": 1.0 if specification["loss"] == "huber" else None
        },
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "change_weighting": weighting,
        "change_weighting_statistics": (
            summarize_change_weighting(train_dataset, weighting)
            if weighting is not None
            else None
        ),
        "batch_size": config.batch_size,
        "seed": config.seed,
        "best_epoch": training["best_epoch"],
        "best_validation_loss": training["best_validation_loss"],
        **summary,
        "checkpoint": str(checkpoint_path),
        "validation_predictions": str(prediction_path),
        "duration_seconds": training["duration_seconds"],
        "device": training["device"],
        "gpu_name": training["gpu_name"],
    }
    log["residual_statistics_train_only"] = residual_statistics
    log["baselines"] = load_validation_baselines()
    log["experiments"].append(experiment)
    log["updated_at_utc"] = datetime.now(UTC).isoformat()
    write_json(LOG_PATH, log)
    print_summary(log)


def calibration_key(item: dict[str, Any]) -> tuple[float, ...]:
    return validation_selection_key(
        {"validation_metrics": item["validation_metrics"]}
    )


def calibrate_experiment(experiment_id: str) -> None:
    if NEW_FINAL_RESULTS.exists():
        raise SystemExit("Final improved test results exist; calibration is closed.")
    log = load_log()
    experiment = next(
        item for item in log["experiments"] if item["id"] == experiment_id
    )
    with np.load(experiment["validation_predictions"]) as values:
        target = values["target"].copy()
        raw_prediction = values["mlp"].copy()
        current = values["persistence"].copy()
        horizon = values["horizon_hours"].copy()

    results: list[dict[str, Any]] = []
    for alpha in (0.25, 0.5, 0.75, 1.0):
        prediction = calibrate_residual_prediction(
            raw_prediction, current, alpha=alpha, deadband=0.0
        )
        results.append(
            {
                "alpha": alpha,
                "deadband": 0.0,
                **summarize_prediction(target, prediction, horizon, current),
            }
        )
    best_alpha = min(results, key=calibration_key)["alpha"]
    for deadband in (0.1, 0.25, 0.5):
        prediction = calibrate_residual_prediction(
            raw_prediction, current, alpha=best_alpha, deadband=deadband
        )
        results.append(
            {
                "alpha": best_alpha,
                "deadband": deadband,
                **summarize_prediction(target, prediction, horizon, current),
            }
        )
    best = min(results, key=calibration_key)
    log["calibration_results"][experiment_id] = {
        "search": (
            "alpha in [0.25, 0.5, 0.75, 1.0] at zero deadband, then "
            "deadband in [0.1, 0.25, 0.5] at the best alpha"
        ),
        "candidates": results,
        "best": best,
    }
    log["updated_at_utc"] = datetime.now(UTC).isoformat()
    write_json(LOG_PATH, log)
    print(
        f"Best calibration for {experiment_id}: alpha={best['alpha']}, "
        f"deadband={best['deadband']}, "
        f"RMSE={best['validation_metrics']['overall']['rmse']:.4f}"
    )


def select_experiment(experiment_id: str) -> None:
    if NEW_FINAL_RESULTS.exists():
        raise SystemExit("Final improved test results already exist.")
    log = load_log()
    log["seed_robustness"] = seed_robustness_summary(log)
    experiment = next(
        item for item in log["experiments"] if item["id"] == experiment_id
    )
    calibration = log["calibration_results"].get(experiment_id)
    best_calibration = calibration["best"] if calibration is not None else {
        "alpha": 1.0,
        "deadband": 0.0,
        **{
            key: experiment[key]
            for key in (
                "validation_metrics",
                "validation_change_status",
                "validation_bias",
            )
        },
    }
    shutil.copy2(experiment["checkpoint"], SELECTED_CHECKPOINT)
    checkpoint = torch.load(SELECTED_CHECKPOINT, map_location="cpu", weights_only=True)
    architecture = checkpoint["architecture"]
    architecture["calibration"] = {
        "alpha": best_calibration["alpha"],
        "deadband": best_calibration["deadband"],
        "order": "deadband raw residual, then multiply by alpha",
    }
    update_checkpoint_metadata(
        SELECTED_CHECKPOINT,
        {
            "architecture": architecture,
            "calibration": architecture["calibration"],
            "change_weighting": experiment["change_weighting"],
            "selection": {
                "uses": "validation only",
                "experiment_id": experiment_id,
                "validation_metrics": best_calibration["validation_metrics"],
                "validation_change_status": best_calibration[
                    "validation_change_status"
                ],
                "validation_bias": best_calibration["validation_bias"],
            },
        },
    )
    log["selected"] = {
        "experiment_id": experiment_id,
        "checkpoint": str(SELECTED_CHECKPOINT),
        "calibration": architecture["calibration"],
        "validation_metrics": best_calibration["validation_metrics"],
        "validation_change_status": best_calibration["validation_change_status"],
        "validation_bias": best_calibration["validation_bias"],
        "selected_at_utc": datetime.now(UTC).isoformat(),
    }
    write_json(LOG_PATH, log)
    print(f"Selected {experiment_id} -> {SELECTED_CHECKPOINT}")


def seed_robustness_summary(log: dict[str, Any]) -> dict[str, Any]:
    experiments = {item["id"]: item for item in log["experiments"]}
    groups = {
        "128_64_mse": ["pc_mse_s42", "pc_mse_s123", "pc_mse_s456"],
        "256_128_64_mse": [
            "pc_large_mse_s42",
            "pc_large_mse_s123",
            "pc_large_mse_s456",
        ],
    }
    result: dict[str, Any] = {}
    for name, ids in groups.items():
        if not all(item in experiments for item in ids):
            continue
        calibration = log["calibration_results"][ids[0]]["best"]
        rows = []
        for item_id in ids:
            experiment = experiments[item_id]
            with np.load(experiment["validation_predictions"]) as values:
                prediction = calibrate_residual_prediction(
                    values["mlp"],
                    values["persistence"],
                    alpha=calibration["alpha"],
                    deadband=calibration["deadband"],
                )
                summary = summarize_prediction(
                    values["target"],
                    prediction,
                    values["horizon_hours"],
                    values["persistence"],
                )
            rows.append(
                {
                    "id": item_id,
                    "seed": experiment["seed"],
                    **summary,
                }
            )
        result[name] = {
            "calibration_fixed_from_seed_42": {
                "alpha": calibration["alpha"],
                "deadband": calibration["deadband"],
            },
            "runs": rows,
            "overall_rmse_mean": float(
                np.mean([row["validation_metrics"]["overall"]["rmse"] for row in rows])
            ),
            "overall_rmse_std": float(
                np.std([row["validation_metrics"]["overall"]["rmse"] for row in rows])
            ),
            "changing_rmse_mean": float(
                np.mean(
                    [
                        row["validation_change_status"]["changing"]["rmse"]
                        for row in rows
                    ]
                )
            ),
            "changing_rmse_std": float(
                np.std(
                    [
                        row["validation_change_status"]["changing"]["rmse"]
                        for row in rows
                    ]
                )
            ),
        }
    return result


def print_summary(log: dict[str, Any]) -> None:
    print("\nValidation experiments (sorted by overall RMSE)")
    print(
        f"{'ID':<28} {'Architecture':<14} {'Loss':<7} {'Weighted':<8} "
        f"{'Seed':>5} {'MAE':>8} {'RMSE':>8} {'R2':>8} {'Unchg':>8} {'Change':>8}"
    )
    for item in sorted(
        log["experiments"], key=lambda row: row["validation_metrics"]["overall"]["rmse"]
    ):
        overall = item["validation_metrics"]["overall"]
        groups = item["validation_change_status"]
        print(
            f"{item['id']:<28} {str(item['architecture']):<14} "
            f"{item['loss']:<7} {str(item['change_weighting'] is not None):<8} "
            f"{item['seed']:>5d} {overall['mae']:>8.4f} {overall['rmse']:>8.4f} "
            f"{overall['r2']:>8.4f} {groups['nearly_unchanged']['rmse']:>8.4f} "
            f"{groups['changing']['rmse']:>8.4f}"
        )


def main() -> None:
    args = parse_args()
    if args.run:
        run_experiment(args.run)
    elif args.calibrate:
        calibrate_experiment(args.calibrate)
    elif args.select:
        select_experiment(args.select)
    else:
        print_summary(load_log())


if __name__ == "__main__":
    main()
