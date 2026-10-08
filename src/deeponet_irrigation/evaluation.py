from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .models import MLPBaseline


def soil_moisture_scaling(metadata: dict[str, Any]) -> tuple[int, float, float]:
    feature_order = list(metadata["feature_order"])
    soil_index = feature_order.index("soil_moisture")
    statistics = metadata["normalization"]["statistics"]["soil_moisture"]
    return soil_index, float(statistics["mean"]), float(statistics["std"])


def load_mlp_checkpoint(
    checkpoint_path: str | Path,
    metadata: dict[str, Any],
    device: torch.device,
) -> tuple[MLPBaseline, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    contract = checkpoint.get("preprocessing_contract")
    expected_contract = {
        "dataset": metadata["dataset"],
        "selected_data_source": metadata["selected_data_source"],
        "selected_sector": metadata["selected_sector"],
        "feature_order": metadata["feature_order"],
        "target_column": metadata["target_column"],
        "history_steps": metadata["history_steps"],
        "prediction_horizons_hours": metadata["prediction_horizons_hours"],
    }
    if contract != expected_contract:
        raise ValueError("Checkpoint preprocessing contract does not match prepared data")
    architecture = checkpoint["architecture"]
    output_scaling = architecture["output_scaling"]
    soil_index, soil_mean, soil_std = soil_moisture_scaling(metadata)
    feature_order = tuple(metadata["feature_order"])
    feature_statistics = metadata["normalization"]["statistics"]
    temporal = architecture.get("temporal_features", {})
    model = MLPBaseline(
        history_steps=int(architecture["history_steps"]),
        feature_count=int(architecture["feature_count"]),
        hidden_sizes=tuple(architecture["hidden_sizes"]),
        target_mean=float(output_scaling["mean"]),
        target_std=float(output_scaling["std"]),
        target_mode=str(architecture.get("target_mode", "direct")),
        soil_feature_index=int(architecture.get("soil_feature_index", soil_index)),
        soil_mean=float(
            architecture.get("soil_moisture_scaling", {}).get("mean", soil_mean)
        ),
        soil_std=float(
            architecture.get("soil_moisture_scaling", {}).get("std", soil_std)
        ),
        residual_scaling_mode=str(
            architecture.get("residual_scaling_mode", "mean_std")
        ),
        output_initialization=str(
            architecture.get("output_initialization", "default")
        ),
        calibration_alpha=float(
            architecture.get("calibration", {}).get("alpha", 1.0)
        ),
        calibration_deadband=float(
            architecture.get("calibration", {}).get("deadband", 0.0)
        ),
        temporal_feature_names=tuple(temporal.get("names", ())),
        temporal_feature_means=tuple(temporal.get("means", ())),
        temporal_feature_stds=tuple(temporal.get("stds", ())),
        input_feature_order=feature_order if temporal.get("names") else (),
        input_feature_means=(
            tuple(feature_statistics[name]["mean"] for name in feature_order)
            if temporal.get("names")
            else ()
        ),
        input_feature_stds=(
            tuple(feature_statistics[name]["std"] for name in feature_order)
            if temporal.get("names")
            else ()
        ),
    )
    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=architecture.get("target_mode") is not None,
    )
    model.to(device)
    return model, checkpoint


def calibrate_residual_prediction(
    prediction: np.ndarray,
    current_soil_moisture: np.ndarray,
    *,
    alpha: float,
    deadband: float,
) -> np.ndarray:
    if alpha < 0 or deadband < 0:
        raise ValueError("Calibration values must be non-negative")
    prediction = np.asarray(prediction)
    current = np.asarray(current_soil_moisture)
    residual = prediction - current
    residual = np.where(np.abs(residual) < deadband, 0.0, residual)
    return current + alpha * residual


def persistence_prediction(
    branch_input: torch.Tensor,
    *,
    soil_index: int,
    soil_mean: float,
    soil_std: float,
) -> torch.Tensor:
    normalized_current = branch_input[:, -1, soil_index]
    return normalized_current.mul(soil_std).add(soil_mean).unsqueeze(1)


def collect_predictions(
    loader: DataLoader[dict[str, torch.Tensor]],
    metadata: dict[str, Any],
    *,
    device: torch.device,
    model: nn.Module | None = None,
) -> dict[str, np.ndarray]:
    soil_index, soil_mean, soil_std = soil_moisture_scaling(metadata)
    output: dict[str, list[np.ndarray]] = {
        "target": [],
        "persistence": [],
        "horizon_hours": [],
        "current_index": [],
        "target_index": [],
    }
    if model is not None:
        output["mlp"] = []
        model.eval()

    with torch.inference_mode():
        for batch in loader:
            branch = batch["branch_input"]
            persistence = persistence_prediction(
                branch,
                soil_index=soil_index,
                soil_mean=soil_mean,
                soil_std=soil_std,
            )
            output["target"].append(batch["target"].numpy())
            output["persistence"].append(persistence.numpy())
            output["horizon_hours"].append(batch["horizon"].numpy())
            output["current_index"].append(batch["current_index"].numpy())
            output["target_index"].append(batch["target_index"].numpy())

            if model is not None:
                prediction = model(
                    branch.to(device, non_blocking=True),
                    batch["horizon"].to(device, non_blocking=True),
                )
                output["mlp"].append(prediction.cpu().numpy())

    return {name: np.concatenate(values) for name, values in output.items()}
