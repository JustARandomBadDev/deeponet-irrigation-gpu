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
CHECKPOINT = ROOT / "models" / "mlp_baseline_final.pt"
SELECTION_LOG = RESULTS_DIR / "mlp_final_feature_experiments.json"
PREVIOUS_TEST = RESULTS_DIR / "final_improved_test_predictions.npz"
FINAL_METRICS = RESULTS_DIR / "final_frozen_mlp_metrics.json"
FINAL_PREDICTIONS = RESULTS_DIR / "final_frozen_mlp_predictions.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the last MLP test evaluation for the frozen baseline"
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


def main() -> None:
    parse_args()
    if FINAL_METRICS.exists() or FINAL_PREDICTIONS.exists():
        raise SystemExit("Final frozen MLP test already exists; refusing to repeat it.")
    selection = json.loads(SELECTION_LOG.read_text(encoding="utf-8"))
    if selection.get("selected", {}).get("checkpoint") != str(CHECKPOINT):
        raise SystemExit("Final validation-only selection is not frozen")

    dataset = PreparedTemporalDataset(DATA_DIR, "test")
    device = select_device()
    model, checkpoint = load_mlp_checkpoint(CHECKPOINT, dataset.metadata, device)
    if not checkpoint.get("final_mlp"):
        raise ValueError("Checkpoint is not marked as the final MLP")
    started = time.perf_counter()
    final = collect_predictions(
        make_data_loader(dataset, batch_size=1024, shuffle=False),
        dataset.metadata,
        device=device,
        model=model,
    )
    inference_seconds = time.perf_counter() - started

    with np.load(PREVIOUS_TEST) as previous:
        for name in ("target", "persistence", "horizon_hours", "current_index", "target_index"):
            if not np.array_equal(final[name], previous[name]):
                raise ValueError(f"Previous and final test arrays differ: {name}")
        previous_prediction = previous["improved_residual_mlp"].copy()

    target = final["target"]
    current = final["persistence"]
    horizon = final["horizon_hours"]
    reports = {
        "persistence": summarize(target, current, horizon, current),
        "previous_best_mlp": summarize(
            target, previous_prediction, horizon, current
        ),
        "final_mlp": summarize(target, final["mlp"], horizon, current),
    }
    np.savez_compressed(
        FINAL_PREDICTIONS,
        target=target,
        persistence=current,
        previous_best_mlp=previous_prediction,
        final_mlp=final["mlp"],
        horizon_hours=horizon,
        current_index=final["current_index"],
        target_index=final["target_index"],
        target_timestamp=np.asarray(dataset.timestamps)[
            final["target_index"].astype(np.int64)
        ],
    )
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "this_is_the_last_mlp_test_evaluation": True,
        "selection_frozen_before_test": True,
        "selection": selection["selected"],
        "test": reports,
        "runtime": {
            "torch_version": torch.__version__,
            "device": str(device),
            "gpu_name": (
                torch.cuda.get_device_name(0) if device.type == "cuda" else None
            ),
            "final_model_inference_seconds": inference_seconds,
        },
    }
    write_json(FINAL_METRICS, report)
    for label, values in reports.items():
        metrics = values["metrics"]["overall"]
        print(
            f"{label:<20} MAE={metrics['mae']:.4f} "
            f"RMSE={metrics['rmse']:.4f} R2={metrics['r2']:.4f} "
            f"unchanged_RMSE={values['change_status']['nearly_unchanged']['rmse']:.4f} "
            f"changing_RMSE={values['change_status']['changing']['rmse']:.4f} "
            f"bias={values['bias']['mean_signed_error']:.4f}"
        )
    print(f"Wrote {FINAL_METRICS}")
    print(f"Wrote {FINAL_PREDICTIONS}")


if __name__ == "__main__":
    main()
