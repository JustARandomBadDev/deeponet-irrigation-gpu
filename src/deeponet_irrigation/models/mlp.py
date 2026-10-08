from __future__ import annotations

import torch
from torch import nn

from ..temporal_features import derive_temporal_features, temporal_feature_metadata


class MLPBaseline(nn.Module):
    def __init__(
        self,
        history_steps: int,
        feature_count: int,
        hidden_sizes: tuple[int, ...] = (128, 64),
        target_mean: float = 0.0,
        target_std: float = 1.0,
        target_mode: str = "direct",
        soil_feature_index: int = 0,
        soil_mean: float = 0.0,
        soil_std: float = 1.0,
        residual_scaling_mode: str = "mean_std",
        output_initialization: str = "default",
        calibration_alpha: float = 1.0,
        calibration_deadband: float = 0.0,
        temporal_feature_names: tuple[str, ...] = (),
        temporal_feature_means: tuple[float, ...] = (),
        temporal_feature_stds: tuple[float, ...] = (),
        input_feature_order: tuple[str, ...] = (),
        input_feature_means: tuple[float, ...] = (),
        input_feature_stds: tuple[float, ...] = (),
    ) -> None:
        super().__init__()
        if target_mode not in {"direct", "residual"}:
            raise ValueError("target_mode must be 'direct' or 'residual'")
        if target_std <= 0 or soil_std <= 0:
            raise ValueError("Scaling standard deviations must be positive")
        if residual_scaling_mode not in {"mean_std", "std_only", "raw"}:
            raise ValueError("Unknown residual scaling mode")
        if output_initialization not in {"default", "zero"}:
            raise ValueError("Unknown output initialization")
        if calibration_alpha < 0 or calibration_deadband < 0:
            raise ValueError("Calibration values must be non-negative")
        if not (
            len(temporal_feature_names)
            == len(temporal_feature_means)
            == len(temporal_feature_stds)
        ):
            raise ValueError("Temporal feature normalization lengths differ")
        if temporal_feature_names and not (
            len(input_feature_order)
            == len(input_feature_means)
            == len(input_feature_stds)
            == feature_count
        ):
            raise ValueError("Input feature scaling is required for temporal features")
        self.history_steps = history_steps
        self.feature_count = feature_count
        self.hidden_sizes = hidden_sizes
        self.target_mode = target_mode
        self.soil_feature_index = soil_feature_index
        self.residual_scaling_mode = residual_scaling_mode
        self.output_initialization = output_initialization
        self.calibration_alpha = float(calibration_alpha)
        self.calibration_deadband = float(calibration_deadband)
        self.temporal_feature_names = temporal_feature_names
        self.input_feature_order = input_feature_order
        self.register_buffer("target_mean", torch.tensor(float(target_mean)))
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
        input_size = history_steps * feature_count + 1 + len(temporal_feature_names)
        layers: list[nn.Module] = []
        previous_size = input_size
        for hidden_size in hidden_sizes:
            layers.extend((nn.Linear(previous_size, hidden_size), nn.ReLU()))
            previous_size = hidden_size
        layers.append(nn.Linear(previous_size, 1))
        self.network = nn.Sequential(*layers)
        if output_initialization == "zero":
            output_layer = self.network[-1]
            assert isinstance(output_layer, nn.Linear)
            nn.init.zeros_(output_layer.weight)
            nn.init.zeros_(output_layer.bias)

    def forward(
        self, branch_input: torch.Tensor, horizon_hours: torch.Tensor
    ) -> torch.Tensor:
        expected = (self.history_steps, self.feature_count)
        if tuple(branch_input.shape[-2:]) != expected:
            raise ValueError(
                f"Expected branch shape [batch, {expected[0]}, {expected[1]}], "
                f"got {tuple(branch_input.shape)}"
            )
        flattened = branch_input.flatten(start_dim=1)
        normalized_horizon = horizon_hours.reshape(-1, 1) / 24.0
        model_inputs = [flattened, normalized_horizon]
        if self.temporal_feature_names:
            derived = derive_temporal_features(
                branch_input,
                self.temporal_feature_names,
                feature_order=self.input_feature_order,
                feature_means=self.input_feature_means,
                feature_stds=self.input_feature_stds,
            )
            normalized_derived = (
                derived - self.temporal_feature_means
            ) / self.temporal_feature_stds
            model_inputs.append(normalized_derived)
        normalized_output = self.network(
            torch.cat(model_inputs, dim=1)
        )
        physical_output = normalized_output * self.target_std + self.target_mean
        if self.target_mode == "direct":
            return physical_output

        normalized_current = branch_input[
            :, -1, self.soil_feature_index : self.soil_feature_index + 1
        ]
        current = normalized_current * self.soil_std + self.soil_mean
        residual = torch.where(
            physical_output.abs() < self.calibration_deadband,
            torch.zeros_like(physical_output),
            physical_output,
        )
        return current + self.calibration_alpha * residual

    def architecture(self) -> dict[str, object]:
        return {
            "history_steps": self.history_steps,
            "feature_count": self.feature_count,
            "hidden_sizes": list(self.hidden_sizes),
            "horizon_encoding": "hours / 24",
            "input_size": self.history_steps * self.feature_count
            + 1
            + len(self.temporal_feature_names),
            "temporal_features": {
                "definitions": temporal_feature_metadata(
                    self.temporal_feature_names
                ),
                "names": list(self.temporal_feature_names),
                "means": self.temporal_feature_means.tolist(),
                "stds": self.temporal_feature_stds.tolist(),
                "normalization_fit_split": "train",
            },
            "target_mode": self.target_mode,
            "residual_scaling_mode": self.residual_scaling_mode,
            "output_initialization": self.output_initialization,
            "calibration": {
                "alpha": self.calibration_alpha,
                "deadband": self.calibration_deadband,
                "order": "deadband raw residual, then multiply by alpha",
            },
            "soil_feature_index": self.soil_feature_index,
            "soil_moisture_scaling": {
                "mean": float(self.soil_mean.item()),
                "std": float(self.soil_std.item()),
                "source": "training-only feature statistics",
            },
            "output_scaling": {
                "mean": float(self.target_mean.item()),
                "std": float(self.target_std.item()),
                "source": (
                    f"training-only residual targets ({self.residual_scaling_mode})"
                    if self.target_mode == "residual"
                    else "training-only soil_moisture statistics"
                ),
            },
        }

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
