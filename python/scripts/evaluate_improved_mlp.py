from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from typing import Any

import numpy as np
import torch

from deeponet_irrigation.dataset import PreparedTemporalDataset, make_data_loader
from deeponet_irrigation.evaluation import collect_predictions, load_mlp_checkpoint
from deeponet_irrigation.metrics import (
    error_by_change_status,
    metrics_by_horizon,
    prediction_bias,
)
from deeponet_irrigation.project_paths import REPOSITORY_ROOT
from deeponet_irrigation.training import select_device, write_json


ROOT = REPOSITORY_ROOT
DATA_DIR = ROOT / "data" / "processed" / "arnesano_v2"
RESULTS_DIR = ROOT / "results"
CHECKPOINT = ROOT / "models" / "mlp_baseline_best.pt"
EXPERIMENT_LOG = RESULTS_DIR / "mlp_improvement_experiments.json"
ORIGINAL_TEST_PREDICTIONS = RESULTS_DIR / "final_test_predictions.npz"
FINAL_METRICS = RESULTS_DIR / "final_improved_baseline_metrics.json"
FINAL_PREDICTIONS = RESULTS_DIR / "final_improved_test_predictions.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the single final test inference for the frozen improved MLP"
    )
    parser.add_argument("--final-test", action="store_true", required=True)
    return parser.parse_args()


def summarize(
    target: np.ndarray,
    prediction: np.ndarray,
    horizon: np.ndarray,
    current: np.ndarray,
) -> dict[str, Any]:
    return {
        "metrics": metrics_by_horizon(target, prediction, horizon),
        "change_status": error_by_change_status(
            target, prediction, current, tolerance=0.5
        ),
        "bias": prediction_bias(target, prediction),
    }


def print_metrics(label: str, report: dict[str, Any]) -> None:
    print(f"\n{label}")
    print(f"{'Horizon':>8} {'N':>7} {'MAE':>10} {'RMSE':>10} {'R2':>10}")
    metrics = report["metrics"]
    for horizon, values in [
        ("overall", metrics["overall"]),
        *metrics["by_horizon"].items(),
    ]:
        print(
            f"{horizon:>8} {values['sample_count']:>7d} "
            f"{values['mae']:>10.4f} {values['rmse']:>10.4f} "
            f"{values['r2']:>10.4f}"
        )
    print(
        "  unchanged RMSE="
        f"{report['change_status']['nearly_unchanged']['rmse']:.4f}; "
        f"changing RMSE={report['change_status']['changing']['rmse']:.4f}; "
        f"bias={report['bias']['mean_signed_error']:.4f}"
    )


def main() -> None:
    parse_args()
    if FINAL_METRICS.exists() or FINAL_PREDICTIONS.exists():
        raise SystemExit("Improved final test artifacts exist; refusing to evaluate again.")
    if not all(
        path.exists()
        for path in (CHECKPOINT, EXPERIMENT_LOG, ORIGINAL_TEST_PREDICTIONS)
    ):
        raise SystemExit("Missing frozen selection or original comparison artifacts.")
    experiment_log = json.loads(EXPERIMENT_LOG.read_text(encoding="utf-8"))
    if "selected" not in experiment_log:
        raise SystemExit("Validation-only model selection has not been frozen.")

    device = select_device()
    dataset = PreparedTemporalDataset(DATA_DIR, "test")
    model, checkpoint = load_mlp_checkpoint(CHECKPOINT, dataset.metadata, device)
    if checkpoint.get("selection", {}).get("uses") != "validation only":
        raise ValueError("Checkpoint does not record validation-only selection")

    started = time.perf_counter()
    improved = collect_predictions(
        make_data_loader(dataset, batch_size=1024, shuffle=False),
        dataset.metadata,
        device=device,
        model=model,
    )
    inference_seconds = time.perf_counter() - started

    with np.load(ORIGINAL_TEST_PREDICTIONS) as original:
        for name in ("target", "persistence", "horizon_hours", "current_index", "target_index"):
            if not np.array_equal(improved[name], original[name]):
                raise ValueError(f"Original and improved test arrays differ: {name}")
        direct_prediction = original["direct_mlp"].copy()
        original_residual_prediction = original["selected_residual_mlp"].copy()

    target = improved["target"]
    current = improved["persistence"]
    horizon = improved["horizon_hours"]
    reports = {
        "persistence": summarize(target, current, horizon, current),
        "original_direct_mlp": summarize(
            target, direct_prediction, horizon, current
        ),
        "original_residual_mlp": summarize(
            target, original_residual_prediction, horizon, current
        ),
        "improved_residual_mlp": summarize(
            target, improved["mlp"], horizon, current
        ),
    }
    np.savez_compressed(
        FINAL_PREDICTIONS,
        target=target,
        persistence=current,
        original_direct_mlp=direct_prediction,
        original_residual_mlp=original_residual_prediction,
        improved_residual_mlp=improved["mlp"],
        horizon_hours=horizon,
        current_index=improved["current_index"],
        target_index=improved["target_index"],
        target_timestamp=np.asarray(dataset.timestamps)[
            improved["target_index"].astype(np.int64)
        ],
    )
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "final_test_evaluation": True,
        "selection_frozen_before_test": True,
        "selected": experiment_log["selected"],
        "seed_robustness": experiment_log["seed_robustness"],
        "test": reports,
        "comparison_artifact": str(ORIGINAL_TEST_PREDICTIONS),
        "runtime": {
            "torch_version": torch.__version__,
            "device": str(device),
            "gpu_name": (
                torch.cuda.get_device_name(0) if device.type == "cuda" else None
            ),
            "improved_model_inference_seconds": inference_seconds,
        },
    }
    write_json(FINAL_METRICS, report)
    for label, values in reports.items():
        print_metrics(label, values)
    print(f"\nWrote {FINAL_METRICS}")
    print(f"Wrote {FINAL_PREDICTIONS}")


if __name__ == "__main__":
    main()
