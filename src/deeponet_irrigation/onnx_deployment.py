from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .evaluation import load_deeponet_checkpoint
from .models import ResidualDeepONet
from .temporal_features import derive_temporal_features


ONNX_INPUT_NAMES = ("branch_input", "horizon", "current_moisture")
ONNX_OUTPUT_NAME = "prediction"
ONNX_OPSET = 17


class DeepONetONNXWrapper(nn.Module):
    """Deployment graph with preprocessing-dependent values as explicit inputs."""

    def __init__(self, model: ResidualDeepONet) -> None:
        super().__init__()
        self.branch_network = model.branch_network
        self.trunk_network = model.trunk_network
        self.operator_bias = model.operator_bias
        self.register_buffer("residual_std", model.target_std.detach().clone())
        self.branch_input_dimension = model.branch_input_dimension

    def forward(
        self,
        branch_input: torch.Tensor,
        horizon: torch.Tensor,
        current_moisture: torch.Tensor,
    ) -> torch.Tensor:
        branch_latent = self.branch_network(branch_input)
        trunk_latent = self.trunk_network(horizon)
        normalized_delta = (
            branch_latent * trunk_latent
        ).sum(dim=-1, keepdim=True) + self.operator_bias
        return current_moisture + normalized_delta * self.residual_std


def load_frozen_deeponet(
    checkpoint_path: str | Path,
    metadata: dict[str, Any],
    device: torch.device = torch.device("cpu"),
) -> tuple[ResidualDeepONet, dict[str, Any]]:
    model, checkpoint = load_deeponet_checkpoint(checkpoint_path, metadata, device)
    if not checkpoint.get("reference_model"):
        raise ValueError("Checkpoint is not marked as the frozen reference model")
    if not checkpoint.get("selection_frozen_before_test"):
        raise ValueError("Checkpoint selection is not marked as frozen")
    model.eval()
    return model, checkpoint


def prepare_onnx_inputs(
    model: ResidualDeepONet,
    historical_input: torch.Tensor,
    horizon_hours: torch.Tensor,
) -> dict[str, np.ndarray]:
    """Convert prepared model histories into the explicit ONNX contract."""
    if historical_input.ndim != 3:
        raise ValueError("historical_input must have shape [batch, history, features]")
    historical_input = historical_input.to(
        device=model.target_std.device, dtype=torch.float32
    )
    horizon_hours = horizon_hours.to(
        device=model.target_std.device, dtype=torch.float32
    ).reshape(-1, 1)
    if historical_input.shape[0] != horizon_hours.shape[0]:
        raise ValueError("Historical input and horizon batch sizes differ")

    branch_parts = [historical_input.flatten(start_dim=1)]
    if model.temporal_feature_names:
        physical_trends = derive_temporal_features(
            historical_input,
            model.temporal_feature_names,
            feature_order=model.feature_order,
            feature_means=model.input_feature_means,
            feature_stds=model.input_feature_stds,
        )
        branch_parts.append(
            (physical_trends - model.temporal_feature_means)
            / model.temporal_feature_stds
        )
    branch_input = torch.cat(branch_parts, dim=1)
    if branch_input.shape[1] != model.branch_input_dimension:
        raise RuntimeError("Prepared Branch input dimension does not match checkpoint")

    normalized_current = historical_input[
        :, -1, model.soil_feature_index : model.soil_feature_index + 1
    ]
    current_moisture = normalized_current * model.soil_std + model.soil_mean
    return {
        "branch_input": branch_input.detach().cpu().numpy().astype(np.float32),
        "horizon": (horizon_hours / 24.0)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32),
        "current_moisture": current_moisture.detach()
        .cpu()
        .numpy()
        .astype(np.float32),
    }


def pytorch_deployment_prediction(
    wrapper: DeepONetONNXWrapper,
    inputs: dict[str, np.ndarray],
) -> np.ndarray:
    device = wrapper.residual_std.device
    with torch.inference_mode():
        prediction = wrapper(
            torch.from_numpy(inputs["branch_input"]).to(device),
            torch.from_numpy(inputs["horizon"]).to(device),
            torch.from_numpy(inputs["current_moisture"]).to(device),
        )
    return prediction.cpu().numpy()


def sample_batch(
    dataset: Any,
    *,
    horizon_hours: int,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    matching = np.flatnonzero(
        np.asarray(dataset.horizon_hours).reshape(-1) == horizon_hours
    )
    if matching.size < batch_size:
        raise ValueError(
            f"Only {matching.size} samples available for {horizon_hours}h"
        )
    positions = np.linspace(0, matching.size - 1, batch_size, dtype=np.int64)
    indices = matching[positions]
    samples = [dataset[int(index)] for index in indices]
    return (
        torch.stack([sample["branch_input"] for sample in samples]),
        torch.stack([sample["horizon"] for sample in samples]),
        torch.stack([sample["target"] for sample in samples]),
        indices,
    )
