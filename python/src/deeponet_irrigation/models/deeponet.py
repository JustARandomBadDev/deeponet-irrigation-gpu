from __future__ import annotations

import torch
from torch import nn

from ..temporal_features import derive_temporal_features, temporal_feature_metadata


def _subnetwork(
    input_dimension: int,
    hidden_dimensions: tuple[int, ...],
    output_dimension: int,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_dimension
    for width in hidden_dimensions:
        layers.extend((nn.Linear(previous, width), nn.ReLU()))
        previous = width
    layers.append(nn.Linear(previous, output_dimension))
    return nn.Sequential(*layers)


class ResidualDeepONet(nn.Module):
    """Basic branch/trunk DeepONet that predicts a physical residual."""

    def __init__(
        self,
        *,
        history_steps: int,
        feature_count: int,
        branch_hidden_dimensions: tuple[int, ...] = (256, 128),
        trunk_hidden_dimensions: tuple[int, ...] = (64, 64),
        latent_dimension: int = 64,
        target_std: float = 1.0,
        soil_feature_index: int = 0,
        soil_mean: float = 0.0,
        soil_std: float = 1.0,
        feature_order: tuple[str, ...] = (),
        sampling_interval_minutes: int = 10,
        temporal_feature_names: tuple[str, ...] = (),
        temporal_feature_means: tuple[float, ...] = (),
        temporal_feature_stds: tuple[float, ...] = (),
        input_feature_means: tuple[float, ...] = (),
        input_feature_stds: tuple[float, ...] = (),
        latent_initialization_std: float = 0.01,
    ) -> None:
        super().__init__()
        if history_steps <= 0 or feature_count <= 0 or latent_dimension <= 0:
            raise ValueError("Model dimensions must be positive")
        if target_std <= 0 or soil_std <= 0:
            raise ValueError("Scaling standard deviations must be positive")
        if latent_initialization_std <= 0:
            raise ValueError("Latent initialization standard deviation must be positive")
        if not (
            len(temporal_feature_names)
            == len(temporal_feature_means)
            == len(temporal_feature_stds)
        ):
            raise ValueError("Temporal feature normalization lengths differ")
        if temporal_feature_names and not (
            len(feature_order)
            == len(input_feature_means)
            == len(input_feature_stds)
            == feature_count
        ):
            raise ValueError("Input feature scaling is required for temporal features")

        self.history_steps = int(history_steps)
        self.feature_count = int(feature_count)
        self.branch_hidden_dimensions = branch_hidden_dimensions
        self.trunk_hidden_dimensions = trunk_hidden_dimensions
        self.latent_dimension = int(latent_dimension)
        self.soil_feature_index = int(soil_feature_index)
        self.feature_order = feature_order
        self.sampling_interval_minutes = int(sampling_interval_minutes)
        self.temporal_feature_names = temporal_feature_names
        self.latent_initialization_std = float(latent_initialization_std)
        self.target_mode = "residual"
        self.residual_scaling_mode = "std_only"

        self.register_buffer("target_mean", torch.tensor(0.0))
        self.register_buffer("target_std", torch.tensor(float(target_std)))
        self.register_buffer("soil_mean", torch.tensor(float(soil_mean)))
        self.register_buffer("soil_std", torch.tensor(float(soil_std)))
        self.register_buffer(
            "temporal_feature_means",
            torch.tensor(temporal_feature_means, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "temporal_feature_stds",
            torch.tensor(temporal_feature_stds, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "input_feature_means",
            torch.tensor(input_feature_means, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "input_feature_stds",
            torch.tensor(input_feature_stds, dtype=torch.float32),
            persistent=False,
        )

        self.branch_input_dimension = (
            self.history_steps * self.feature_count + len(temporal_feature_names)
        )
        self.trunk_input_dimension = 1
        self.branch_network = _subnetwork(
            self.branch_input_dimension,
            branch_hidden_dimensions,
            self.latent_dimension,
        )
        self.trunk_network = _subnetwork(
            self.trunk_input_dimension,
            trunk_hidden_dimensions,
            self.latent_dimension,
        )
        self.operator_bias = nn.Parameter(torch.zeros(1))
        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        for network in (self.branch_network, self.trunk_network):
            linear_layers = [layer for layer in network if isinstance(layer, nn.Linear)]
            for layer in linear_layers[:-1]:
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)
            nn.init.normal_(
                linear_layers[-1].weight,
                mean=0.0,
                std=self.latent_initialization_std,
            )
            nn.init.zeros_(linear_layers[-1].bias)

    def encode_branch(self, branch_input: torch.Tensor) -> torch.Tensor:
        expected = (self.history_steps, self.feature_count)
        if branch_input.ndim != 3 or tuple(branch_input.shape[-2:]) != expected:
            raise ValueError(
                f"Expected branch shape [batch, {expected[0]}, {expected[1]}], "
                f"got {tuple(branch_input.shape)}"
            )
        inputs = [branch_input.flatten(start_dim=1)]
        if self.temporal_feature_names:
            derived = derive_temporal_features(
                branch_input,
                self.temporal_feature_names,
                feature_order=self.feature_order,
                feature_means=self.input_feature_means,
                feature_stds=self.input_feature_stds,
            )
            inputs.append(
                (derived - self.temporal_feature_means) / self.temporal_feature_stds
            )
        return self.branch_network(torch.cat(inputs, dim=1))

    def encode_trunk(self, horizon_hours: torch.Tensor) -> torch.Tensor:
        normalized_horizon = horizon_hours.reshape(-1, 1) / 24.0
        return self.trunk_network(normalized_horizon)

    def combine_latents(
        self, branch_latent: torch.Tensor, trunk_latent: torch.Tensor
    ) -> torch.Tensor:
        expected = (branch_latent.shape[0], self.latent_dimension)
        if tuple(branch_latent.shape) != expected:
            raise ValueError(f"Unexpected branch latent shape {tuple(branch_latent.shape)}")
        if tuple(trunk_latent.shape) != expected:
            raise ValueError(f"Unexpected trunk latent shape {tuple(trunk_latent.shape)}")
        return (branch_latent * trunk_latent).sum(dim=-1, keepdim=True) + self.operator_bias

    def normalized_residual(
        self, branch_input: torch.Tensor, horizon_hours: torch.Tensor
    ) -> torch.Tensor:
        return self.combine_latents(
            self.encode_branch(branch_input), self.encode_trunk(horizon_hours)
        )

    def forward(
        self, branch_input: torch.Tensor, horizon_hours: torch.Tensor
    ) -> torch.Tensor:
        normalized_delta = self.normalized_residual(branch_input, horizon_hours)
        physical_delta = normalized_delta * self.target_std
        normalized_current = branch_input[
            :, -1, self.soil_feature_index : self.soil_feature_index + 1
        ]
        current = normalized_current * self.soil_std + self.soil_mean
        return current + physical_delta

    def architecture(self) -> dict[str, object]:
        return {
            "model_type": "residual_deeponet",
            "history_steps": self.history_steps,
            "feature_count": self.feature_count,
            "feature_order": list(self.feature_order),
            "sampling_interval_minutes": self.sampling_interval_minutes,
            "branch_input_dimension": self.branch_input_dimension,
            "branch_hidden_dimensions": list(self.branch_hidden_dimensions),
            "trunk_input_dimension": self.trunk_input_dimension,
            "trunk_hidden_dimensions": list(self.trunk_hidden_dimensions),
            "latent_dimension": self.latent_dimension,
            "combination": "sum(branch_latent * trunk_latent) + learnable_bias",
            "horizon_normalization": "horizon_hours / 24",
            "target_mode": self.target_mode,
            "residual_definition": "future_soil_moisture - current_soil_moisture",
            "residual_scaling_mode": self.residual_scaling_mode,
            "residual_scaling_rule": "scaled_delta = delta / train_residual_std",
            "train_residual_std": float(self.target_std.item()),
            "latent_initialization_std": self.latent_initialization_std,
            "temporal_features": {
                "included": bool(self.temporal_feature_names),
                "definitions": temporal_feature_metadata(self.temporal_feature_names),
                "names": list(self.temporal_feature_names),
                "means": self.temporal_feature_means.tolist(),
                "stds": self.temporal_feature_stds.tolist(),
                "normalization_fit_split": "train",
            },
            "soil_feature_index": self.soil_feature_index,
            "soil_moisture_scaling": {
                "mean": float(self.soil_mean.item()),
                "std": float(self.soil_std.item()),
                "source": "training-only feature statistics",
            },
            "output_scaling": {
                "mean": 0.0,
                "std": float(self.target_std.item()),
                "source": "training-only residual targets (std_only)",
            },
        }

    def parameter_counts(self) -> dict[str, int]:
        branch = sum(parameter.numel() for parameter in self.branch_network.parameters())
        trunk = sum(parameter.numel() for parameter in self.trunk_network.parameters())
        return {
            "branch": branch,
            "trunk": trunk,
            "operator_bias": self.operator_bias.numel(),
            "total": branch + trunk + self.operator_bias.numel(),
        }

    def trainable_parameter_count(self) -> int:
        return self.parameter_counts()["total"]
