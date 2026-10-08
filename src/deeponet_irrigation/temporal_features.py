from __future__ import annotations

from typing import Any

import torch


TEMPORAL_FEATURE_SPECS: dict[str, dict[str, Any]] = {
    "soil_moisture_trend_1h": {
        "kind": "trend",
        "source": "soil_moisture",
        "steps": 6,
    },
    "soil_moisture_trend_3h": {
        "kind": "trend",
        "source": "soil_moisture",
        "steps": 18,
    },
    "soil_moisture_trend_6h": {
        "kind": "trend",
        "source": "soil_moisture",
        "steps": 36,
    },
    "irrigation_sum_1h": {
        "kind": "sum",
        "source": "applied_water_liters",
        "steps": 6,
    },
    "irrigation_sum_6h": {
        "kind": "sum",
        "source": "applied_water_liters",
        "steps": 36,
    },
    "irrigation_sum_24h": {
        "kind": "sum",
        "source": "applied_water_liters",
        "steps": 144,
    },
    "rain_sum_1h": {
        "kind": "sum",
        "source": "weather_rain",
        "steps": 6,
    },
    "rain_sum_6h": {
        "kind": "sum",
        "source": "weather_rain",
        "steps": 36,
    },
    "rain_sum_24h": {
        "kind": "sum",
        "source": "weather_rain",
        "steps": 144,
    },
}

TREND_FEATURES = (
    "soil_moisture_trend_1h",
    "soil_moisture_trend_3h",
    "soil_moisture_trend_6h",
)

WATER_FEATURES = (
    "irrigation_sum_1h",
    "irrigation_sum_6h",
    "irrigation_sum_24h",
    "rain_sum_1h",
    "rain_sum_6h",
    "rain_sum_24h",
)


def derive_temporal_features(
    branch_input: torch.Tensor,
    names: tuple[str, ...],
    *,
    feature_order: tuple[str, ...],
    feature_means: torch.Tensor,
    feature_stds: torch.Tensor,
) -> torch.Tensor:
    """Derive physical features using only rows at or before the origin."""
    if not names:
        return branch_input.new_empty((branch_input.shape[0], 0))
    physical = branch_input * feature_stds.view(1, 1, -1) + feature_means.view(
        1, 1, -1
    )
    output = []
    for name in names:
        spec = TEMPORAL_FEATURE_SPECS[name]
        source_index = feature_order.index(spec["source"])
        steps = int(spec["steps"])
        if steps > branch_input.shape[1]:
            raise ValueError(f"History is too short for {name}")
        source = physical[:, :, source_index]
        if spec["kind"] == "trend":
            if steps >= branch_input.shape[1]:
                raise ValueError(f"History is too short for {name}")
            value = source[:, -1] - source[:, -1 - steps]
        else:
            value = source[:, -steps:].sum(dim=1)
        output.append(value)
    return torch.stack(output, dim=1)


def temporal_feature_metadata(names: tuple[str, ...]) -> list[dict[str, Any]]:
    metadata = []
    for name in names:
        spec = TEMPORAL_FEATURE_SPECS[name]
        steps = int(spec["steps"])
        metadata.append(
            {
                "name": name,
                **spec,
                "latest_offset_steps": 0,
                "earliest_offset_steps": (
                    -steps if spec["kind"] == "trend" else 1 - steps
                ),
                "uses_future": False,
            }
        )
    return metadata
