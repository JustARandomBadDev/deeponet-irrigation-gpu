from __future__ import annotations

import numpy as np


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if target.shape != prediction.shape or target.size == 0:
        raise ValueError("Target and prediction must be non-empty arrays of equal shape")

    error = prediction - target
    residual_sum_squares = float(np.sum(error**2))
    total_sum_squares = float(np.sum((target - target.mean()) ** 2))
    if total_sum_squares:
        r2 = 1.0 - residual_sum_squares / total_sum_squares
    else:
        r2 = 1.0 if residual_sum_squares == 0.0 else 0.0
    return {
        "sample_count": int(target.size),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "r2": r2,
    }


def metrics_by_horizon(
    target: np.ndarray,
    prediction: np.ndarray,
    horizon_hours: np.ndarray,
) -> dict[str, object]:
    target = np.asarray(target).reshape(-1)
    prediction = np.asarray(prediction).reshape(-1)
    horizon_hours = np.asarray(horizon_hours).reshape(-1)
    if not (target.shape == prediction.shape == horizon_hours.shape):
        raise ValueError("Targets, predictions, and horizons must have equal shape")

    by_horizon: dict[str, dict[str, float | int]] = {}
    for horizon in sorted(np.unique(horizon_hours)):
        mask = horizon_hours == horizon
        by_horizon[f"{int(horizon)}h"] = regression_metrics(
            target[mask], prediction[mask]
        )
    return {
        "overall": regression_metrics(target, prediction),
        "by_horizon": by_horizon,
    }


def plateau_statistics(
    target: np.ndarray,
    current_soil_moisture: np.ndarray,
    horizon_hours: np.ndarray,
    *,
    tolerance: float = 0.5,
) -> dict[str, object]:
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    current = np.asarray(current_soil_moisture, dtype=np.float64).reshape(-1)
    horizons = np.asarray(horizon_hours).reshape(-1)
    if not (target.shape == current.shape == horizons.shape):
        raise ValueError("Plateau inputs must have equal shape")

    exact_epsilon = 1e-5
    result: dict[str, object] = {
        "unchanged_epsilon": exact_epsilon,
        "tolerance": tolerance,
        "by_horizon": {},
    }
    for horizon in sorted(np.unique(horizons)):
        mask = horizons == horizon
        absolute_change = np.abs(target[mask] - current[mask])
        result["by_horizon"][f"{int(horizon)}h"] = {
            "sample_count": int(mask.sum()),
            "unchanged_fraction": float(np.mean(absolute_change <= exact_epsilon)),
            "within_tolerance_fraction": float(np.mean(absolute_change <= tolerance)),
            "mean_absolute_target_change": float(np.mean(absolute_change)),
            "persistence_mae": float(np.mean(absolute_change)),
        }
    return result


def error_by_change_status(
    target: np.ndarray,
    prediction: np.ndarray,
    current_soil_moisture: np.ndarray,
    *,
    tolerance: float = 0.5,
) -> dict[str, dict[str, float | int]]:
    target = np.asarray(target).reshape(-1)
    prediction = np.asarray(prediction).reshape(-1)
    current = np.asarray(current_soil_moisture).reshape(-1)
    unchanged = np.abs(target - current) <= tolerance
    return {
        "nearly_unchanged": regression_metrics(target[unchanged], prediction[unchanged]),
        "changing": regression_metrics(target[~unchanged], prediction[~unchanged]),
    }


def prediction_bias(
    target: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    return {
        "mean_prediction": float(prediction.mean()),
        "mean_target": float(target.mean()),
        "mean_signed_error": float(np.mean(prediction - target)),
    }
