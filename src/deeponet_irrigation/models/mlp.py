from __future__ import annotations

import torch
from torch import nn


class MLPBaseline(nn.Module):
    def __init__(
        self,
        history_steps: int,
        feature_count: int,
        hidden_sizes: tuple[int, int] = (128, 64),
        target_mean: float = 0.0,
        target_std: float = 1.0,
    ) -> None:
        super().__init__()
        self.history_steps = history_steps
        self.feature_count = feature_count
        self.hidden_sizes = hidden_sizes
        self.register_buffer("target_mean", torch.tensor(float(target_mean)))
        self.register_buffer("target_std", torch.tensor(float(target_std)))
        input_size = history_steps * feature_count + 1
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1),
        )

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
        normalized_prediction = self.network(
            torch.cat((flattened, normalized_horizon), dim=1)
        )
        return normalized_prediction * self.target_std + self.target_mean

    def architecture(self) -> dict[str, object]:
        return {
            "history_steps": self.history_steps,
            "feature_count": self.feature_count,
            "hidden_sizes": list(self.hidden_sizes),
            "horizon_encoding": "hours / 24",
            "output_scaling": {
                "mean": float(self.target_mean.item()),
                "std": float(self.target_std.item()),
                "source": "training-only soil_moisture statistics",
            },
        }

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
