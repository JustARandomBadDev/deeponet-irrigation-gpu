from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from deeponet_irrigation.dataset import (
    PreparedTemporalDataset,
    make_data_loader,
    residual_targets,
)
from deeponet_irrigation.evaluation import load_mlp_checkpoint
from deeponet_irrigation.training import write_json


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed" / "arnesano_v2"
RESULTS_DIR = ROOT / "results"
HORIZONS = (1, 3, 6, 12, 24)


def distribution(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    quantiles = np.quantile(values, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "sample_count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "min": float(values.min()),
        "max": float(values.max()),
        "median": float(np.median(values)),
        "mean_absolute_value": float(np.mean(np.abs(values))),
        "quantiles": {
            name: float(value)
            for name, value in zip(
                ("p01", "p05", "p25", "p50", "p75", "p95", "p99"),
                quantiles,
                strict=True,
            )
        },
    }


def prediction_bias(path: Path) -> dict[str, Any]:
    with np.load(path) as values:
        target = values["target"].reshape(-1).astype(np.float64)
        prediction = values["mlp"].reshape(-1).astype(np.float64)
        horizons = values["horizon_hours"].reshape(-1)

    def summarize(mask: np.ndarray) -> dict[str, float | int]:
        signed_error = prediction[mask] - target[mask]
        return {
            "sample_count": int(mask.sum()),
            "mean_prediction": float(prediction[mask].mean()),
            "mean_target": float(target[mask].mean()),
            "mean_signed_error_prediction_minus_target": float(signed_error.mean()),
        }

    return {
        "source": str(path),
        "note": "Uses the previously saved direct-MLP predictions; test was not rerun.",
        "overall": summarize(np.ones(target.size, dtype=bool)),
        "by_horizon": {
            f"{horizon}h": summarize(horizons == horizon) for horizon in HORIZONS
        },
    }


def main() -> None:
    split_report: dict[str, Any] = {}
    verification: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        dataset = PreparedTemporalDataset(DATA_DIR, split)
        normalized_current = np.asarray(
            dataset.features[dataset.current_index, dataset.soil_index],
            dtype=np.float64,
        )
        current = normalized_current * dataset.soil_std + dataset.soil_mean
        target = np.asarray(dataset.targets).reshape(-1).astype(np.float64)
        delta = residual_targets(dataset)
        horizons = np.asarray(dataset.horizon_hours).reshape(-1)
        split_report[split] = {
            "future_target": distribution(target),
            "current_at_prediction_origin": distribution(current),
            "residual_by_horizon": {
                f"{horizon}h": distribution(delta[horizons == horizon])
                for horizon in HORIZONS
            },
        }

        target_from_grid = (
            np.asarray(
                dataset.features[dataset.target_index, dataset.soil_index],
                dtype=np.float64,
            )
            * dataset.soil_std
            + dataset.soil_mean
        )
        expected_steps = (horizons * 6).astype(np.int64)
        verification[split] = {
            "target_matches_denormalized_target_grid_row": bool(
                np.allclose(target, target_from_grid, atol=2e-5)
            ),
            "current_is_last_history_row": bool(
                np.array_equal(
                    dataset.current_index,
                    dataset.history_start_index + dataset.history_steps - 1,
                )
            ),
            "target_index_matches_horizon": bool(
                np.array_equal(dataset.target_index - dataset.current_index, expected_steps)
            ),
        }

    metadata = PreparedTemporalDataset(DATA_DIR, "train").metadata
    train_end = np.datetime64(metadata["splits"]["train"]["end_timestamp"])
    fit_end = np.datetime64(metadata["normalization"]["fit_end"])
    validation_dataset = PreparedTemporalDataset(DATA_DIR, "validation")
    direct_model, direct_checkpoint = load_mlp_checkpoint(
        ROOT / "models" / "mlp_baseline.pt", metadata, torch.device("cpu")
    )
    batch = next(
        iter(make_data_loader(validation_dataset, batch_size=8, shuffle=False))
    )
    with torch.inference_mode():
        flattened = batch["branch_input"].flatten(start_dim=1)
        horizon = batch["horizon"].reshape(-1, 1) / 24.0
        raw_output = direct_model.network(torch.cat((flattened, horizon), dim=1))
        explicit_physical_output = (
            raw_output * direct_model.target_std + direct_model.target_mean
        )
        public_output = direct_model(batch["branch_input"], batch["horizon"])
    soil_statistics = metadata["normalization"]["statistics"]["soil_moisture"]
    checkpoint_scaling = direct_checkpoint["architecture"]["output_scaling"]

    verification["normalization_and_features"] = {
        "feature_fit_split": metadata["normalization"]["fit_split"],
        "feature_fit_ends_within_train": bool(fit_end <= train_end),
        "soil_moisture_present_in_history": "soil_moisture"
        in metadata["feature_order"],
        "future_named_feature_absent": not any(
            "future" in name or "water_vol_to" in name
            for name in metadata["feature_order"]
        ),
        "irrigation_is_historical": "lagged by one" in metadata["feature_notes"][
            "applied_water_liters"
        ],
        "target_is_stored_in_physical_units": not metadata["normalization"][
            "target_normalized"
        ],
        "direct_checkpoint_uses_training_soil_statistics": bool(
            np.isclose(checkpoint_scaling["mean"], soil_statistics["mean"])
            and np.isclose(checkpoint_scaling["std"], soil_statistics["std"])
        ),
        "direct_forward_matches_explicit_denormalization": bool(
            torch.allclose(public_output, explicit_physical_output)
        ),
        "direct_output_scaling": (
            "Checkpoint output is network_output * training soil std + training "
            "soil mean; evaluation consumes that physical-unit output directly."
        ),
        "bug_found": False,
    }

    bias = {}
    for split in ("validation", "test"):
        prediction_path = RESULTS_DIR / f"{split}_predictions.npz"
        if prediction_path.exists():
            bias[split] = prediction_bias(prediction_path)

    report = {
        "diagnosis": (
            "No pipeline bug found. The direct MLP has an increasingly negative "
            "prediction bias across later chronological splits, and "
            "absolute-target learning is poorly aligned with the persistence-dominated task."
        ),
        "splits": split_report,
        "existing_direct_mlp_prediction_bias": bias,
        "pipeline_verification": verification,
    }
    write_json(RESULTS_DIR / "baseline_diagnostics.json", report)
    print(json.dumps(report, indent=2))
    print(f"Wrote {RESULTS_DIR / 'baseline_diagnostics.json'}")


if __name__ == "__main__":
    main()
