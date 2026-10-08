from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from deeponet_irrigation.dataset import (
    PreparedTemporalDataset,
    fit_residual_normalization,
    fit_temporal_feature_normalization,
    make_data_loader,
)
from deeponet_irrigation.evaluation import collect_predictions, load_deeponet_checkpoint
from deeponet_irrigation.metrics import (
    error_by_change_status,
    metrics_by_horizon,
    prediction_bias,
)
from deeponet_irrigation.models import ResidualDeepONet
from deeponet_irrigation.temporal_features import TREND_FEATURES
from deeponet_irrigation.training import (
    TrainingConfig,
    select_device,
    set_deterministic_seed,
    train_model,
    update_checkpoint_metadata,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed" / "arnesano_v2"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
LOG_PATH = RESULTS_DIR / "deeponet_experiments.json"
FINAL_CHECKPOINT = MODELS_DIR / "deeponet_reference.pt"
FINAL_TEST_RESULTS = RESULTS_DIR / "deeponet_final_test_metrics.json"
PREDICTIONS_DIR = RESULTS_DIR / "deeponet_validation_predictions"
MAX_RUNS = 6


EXPERIMENTS: dict[str, dict[str, Any]] = {
    "initial_s42": {
        "configuration": "initial_raw",
        "branch": (256, 128),
        "trunk": (64, 64),
        "latent": 64,
        "learning_rate": 1e-3,
        "features": (),
        "seed": 42,
    },
    "small_trunk_s42": {
        "configuration": "small_trunk_raw",
        "branch": (256, 128),
        "trunk": (32, 32),
        "latent": 64,
        "learning_rate": 1e-3,
        "features": (),
        "seed": 42,
    },
    "low_lr_s42": {
        "configuration": "initial_raw_low_lr",
        "branch": (256, 128),
        "trunk": (64, 64),
        "latent": 64,
        "learning_rate": 3e-4,
        "features": (),
        "seed": 42,
    },
    "latent128_s42": {
        "configuration": "latent128_raw",
        "branch": (256, 128),
        "trunk": (64, 64),
        "latent": 128,
        "learning_rate": 1e-3,
        "features": (),
        "seed": 42,
    },
    "trends_s42": {
        "configuration": "initial_trends",
        "branch": (256, 128),
        "trunk": (64, 64),
        "latent": 64,
        "learning_rate": 1e-3,
        "features": TREND_FEATURES,
        "seed": 42,
    },
    "final_s123": {
        "configuration": "initial_trends",
        "branch": (256, 128),
        "trunk": (64, 64),
        "latent": 64,
        "learning_rate": 1e-3,
        "features": TREND_FEATURES,
        "seed": 123,
    },
    "final_s456": {
        "configuration": "initial_trends",
        "branch": (256, 128),
        "trunk": (64, 64),
        "latent": 64,
        "learning_rate": 1e-3,
        "features": TREND_FEATURES,
        "seed": 456,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the bounded DeepONet reference")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--run", choices=tuple(EXPERIMENTS))
    action.add_argument("--select", metavar="SEED_42_EXPERIMENT_ID")
    action.add_argument("--summary", action="store_true")
    return parser.parse_args()


def load_validation_baselines() -> dict[str, Any]:
    improvement = json.loads(
        (RESULTS_DIR / "mlp_improvement_experiments.json").read_text(
            encoding="utf-8"
        )
    )
    frozen_mlp = json.loads(
        (RESULTS_DIR / "mlp_final_feature_experiments.json").read_text(
            encoding="utf-8"
        )
    )["selected"]
    return {
        "persistence": improvement["baselines"]["persistence"],
        "frozen_mlp": {
            "metrics": frozen_mlp["validation_metrics"],
            "change_status": frozen_mlp["validation_change_status"],
            "bias": frozen_mlp["validation_bias"],
        },
    }


def load_log() -> dict[str, Any]:
    if LOG_PATH.exists():
        return json.loads(LOG_PATH.read_text(encoding="utf-8"))
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "selection_uses": "train and validation only; test data and metrics are not loaded",
        "maximum_training_runs": MAX_RUNS,
        "baselines": load_validation_baselines(),
        "experiments": [],
    }


def input_scaling(metadata: dict[str, Any]) -> tuple[tuple[str, ...], tuple[float, ...], tuple[float, ...]]:
    order = tuple(metadata["feature_order"])
    statistics = metadata["normalization"]["statistics"]
    means = tuple(float(statistics[name]["mean"]) for name in order)
    stds = tuple(float(statistics[name]["std"]) for name in order)
    return order, means, stds


def build_model(
    specification: dict[str, Any],
    dataset: PreparedTemporalDataset,
    residual_std: float,
    temporal: dict[str, Any],
) -> ResidualDeepONet:
    order, means, stds = input_scaling(dataset.metadata)
    return ResidualDeepONet(
        history_steps=dataset.history_steps,
        feature_count=dataset.feature_count,
        branch_hidden_dimensions=specification["branch"],
        trunk_hidden_dimensions=specification["trunk"],
        latent_dimension=specification["latent"],
        target_std=residual_std,
        soil_feature_index=dataset.soil_index,
        soil_mean=dataset.soil_mean,
        soil_std=dataset.soil_std,
        feature_order=order,
        sampling_interval_minutes=int(dataset.metadata["sampling_interval_minutes"]),
        temporal_feature_names=specification["features"],
        temporal_feature_means=tuple(temporal.get("means", ())),
        temporal_feature_stds=tuple(temporal.get("stds", ())),
        input_feature_means=means if specification["features"] else (),
        input_feature_stds=stds if specification["features"] else (),
    )


def initial_output_diagnostic(
    model: ResidualDeepONet,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, float]:
    batch = next(iter(loader))
    branch = batch["branch_input"].to(device)
    horizon = batch["horizon"].to(device)
    model.to(device).eval()
    with torch.inference_mode():
        normalized = model.normalized_residual(branch, horizon).cpu().numpy()
    physical = normalized * float(model.target_std.item())
    diagnostic = {
        "normalized_mean": float(normalized.mean()),
        "normalized_std": float(normalized.std()),
        "physical_delta_mean": float(physical.mean()),
        "physical_delta_std": float(physical.std()),
        "physical_delta_min": float(physical.min()),
        "physical_delta_max": float(physical.max()),
    }
    print("Initial residual diagnostic:", json.dumps(diagnostic, indent=2))
    return diagnostic


def summarize_predictions(predictions: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        "validation_metrics": metrics_by_horizon(
            predictions["target"],
            predictions["deeponet"],
            predictions["horizon_hours"],
        ),
        "validation_change_status": error_by_change_status(
            predictions["target"],
            predictions["deeponet"],
            predictions["persistence"],
            tolerance=0.5,
        ),
        "validation_bias": prediction_bias(
            predictions["target"], predictions["deeponet"]
        ),
    }


def run_experiment(experiment_id: str) -> None:
    if FINAL_TEST_RESULTS.exists():
        raise SystemExit("The final DeepONet test exists; training is permanently closed.")
    log = load_log()
    if len(log["experiments"]) >= MAX_RUNS:
        raise SystemExit("The six-run DeepONet limit has been reached.")
    if any(item["id"] == experiment_id for item in log["experiments"]):
        raise SystemExit(f"Experiment {experiment_id} already exists")

    specification = EXPERIMENTS[experiment_id]
    train_dataset = PreparedTemporalDataset(DATA_DIR, "train")
    validation_dataset = PreparedTemporalDataset(DATA_DIR, "validation")
    residual = fit_residual_normalization(train_dataset)
    temporal = (
        fit_temporal_feature_normalization(train_dataset, specification["features"])
        if specification["features"]
        else {"names": [], "means": [], "stds": [], "fit_split": "train"}
    )
    config = TrainingConfig(
        batch_size=512,
        learning_rate=specification["learning_rate"],
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
    model = build_model(
        specification, train_dataset, float(residual["std"]), temporal
    )
    device = select_device()
    counts = model.parameter_counts()
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(
        "GPU:", torch.cuda.get_device_name(0) if device.type == "cuda" else "none (CPU)"
    )
    print(f"Branch: {model.branch_input_dimension} -> {specification['branch']} -> {specification['latent']}")
    print(f"Trunk: 1 -> {specification['trunk']} -> {specification['latent']}")
    print(f"Parameters: branch={counts['branch']}, trunk={counts['trunk']}, total={counts['total']}")
    diagnostic = initial_output_diagnostic(model, validation_loader, device)

    checkpoint_path = MODELS_DIR / "deeponet_experiments" / f"{experiment_id}.pt"
    training = train_model(
        model,
        train_loader,
        validation_loader,
        device=device,
        config=config,
        checkpoint_path=checkpoint_path,
        preprocessing_metadata=train_dataset.metadata,
    )
    model, _ = load_deeponet_checkpoint(
        checkpoint_path, train_dataset.metadata, device
    )
    predictions = collect_predictions(
        validation_loader,
        validation_dataset.metadata,
        device=device,
        model=model,
        prediction_name="deeponet",
    )
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    prediction_path = PREDICTIONS_DIR / f"{experiment_id}.npz"
    np.savez_compressed(prediction_path, **predictions)
    experiment = {
        "id": experiment_id,
        "configuration": specification["configuration"],
        "branch_hidden_dimensions": list(specification["branch"]),
        "trunk_hidden_dimensions": list(specification["trunk"]),
        "latent_dimension": specification["latent"],
        "trend_features": list(specification["features"]),
        "feature_normalization": temporal,
        "learning_rate": specification["learning_rate"],
        "loss": "mse",
        "batch_size": config.batch_size,
        "seed": config.seed,
        "parameter_counts": counts,
        "total_parameters": counts["total"],
        "initial_output_diagnostic": diagnostic,
        "best_epoch": training["best_epoch"],
        "best_validation_loss": training["best_validation_loss"],
        "epochs_completed": training["epochs_completed"],
        "duration_seconds": training["duration_seconds"],
        "device": training["device"],
        "gpu_name": training["gpu_name"],
        **summarize_predictions(predictions),
        "checkpoint": str(checkpoint_path),
        "validation_predictions": str(prediction_path),
    }
    log["experiments"].append(experiment)
    log["updated_at_utc"] = datetime.now(UTC).isoformat()
    write_json(LOG_PATH, log)
    print_summary(log)


def select_final(identifier: str) -> None:
    if FINAL_TEST_RESULTS.exists():
        raise SystemExit("The final DeepONet test exists; selection is closed.")
    log = load_log()
    selected = next(item for item in log["experiments"] if item["id"] == identifier)
    if selected["seed"] != 42:
        raise ValueError("Freeze the seed-42 member of the selected configuration")
    matching = [
        item
        for item in log["experiments"]
        if item["configuration"] == selected["configuration"]
        and item["trend_features"] == selected["trend_features"]
    ]
    if {item["seed"] for item in matching} != {42, 123, 456}:
        raise ValueError("Seed robustness runs 42, 123, and 456 must finish first")
    overall = [item["validation_metrics"]["overall"]["rmse"] for item in matching]
    changing = [
        item["validation_change_status"]["changing"]["rmse"] for item in matching
    ]
    robustness = {
        "seeds": [item["seed"] for item in matching],
        "overall_rmse_values": overall,
        "overall_rmse_mean": float(np.mean(overall)),
        "overall_rmse_std": float(np.std(overall)),
        "changing_rmse_values": changing,
        "changing_rmse_mean": float(np.mean(changing)),
        "changing_rmse_std": float(np.std(changing)),
    }
    FINAL_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(selected["checkpoint"], FINAL_CHECKPOINT)
    selection = {
        "uses": "validation only",
        "id": identifier,
        "configuration": selected["configuration"],
        "seed": selected["seed"],
        "validation_metrics": selected["validation_metrics"],
        "validation_change_status": selected["validation_change_status"],
        "validation_bias": selected["validation_bias"],
        "seed_robustness": robustness,
        "selected_at_utc": datetime.now(UTC).isoformat(),
    }
    update_checkpoint_metadata(
        FINAL_CHECKPOINT,
        {"reference_model": True, "selection_frozen_before_test": True, "final_selection": selection},
    )
    log["selected"] = {"checkpoint": str(FINAL_CHECKPOINT), **selection}
    write_json(LOG_PATH, log)
    print(f"DeepONet reference frozen: {identifier} -> {FINAL_CHECKPOINT}")


def print_summary(log: dict[str, Any]) -> None:
    print("\nDeepONet validation experiments")
    print(
        f"{'ID':<20} {'Branch':>12} {'Trunk':>10} {'Latent':>7} "
        f"{'Seed':>5} {'MAE':>8} {'RMSE':>8} {'R2':>8} {'12h':>8} "
        f"{'24h':>8} {'Change':>8} {'Params':>9}"
    )
    for item in log["experiments"]:
        metrics = item["validation_metrics"]
        print(
            f"{item['id']:<20} {str(item['branch_hidden_dimensions']):>12} "
            f"{str(item['trunk_hidden_dimensions']):>10} "
            f"{item['latent_dimension']:>7d} {item['seed']:>5d} "
            f"{metrics['overall']['mae']:>8.4f} {metrics['overall']['rmse']:>8.4f} "
            f"{metrics['overall']['r2']:>8.4f} "
            f"{metrics['by_horizon']['12h']['rmse']:>8.4f} "
            f"{metrics['by_horizon']['24h']['rmse']:>8.4f} "
            f"{item['validation_change_status']['changing']['rmse']:>8.4f} "
            f"{item['total_parameters']:>9d}"
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
