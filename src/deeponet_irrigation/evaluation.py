from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def soil_moisture_scaling(metadata: dict[str, Any]) -> tuple[int, float, float]:
    feature_order = list(metadata["feature_order"])
    soil_index = feature_order.index("soil_moisture")
    statistics = metadata["normalization"]["statistics"]["soil_moisture"]
    return soil_index, float(statistics["mean"]), float(statistics["std"])


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
