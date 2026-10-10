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
from deeponet_irrigation.metrics import error_by_change_status, metrics_by_horizon
from deeponet_irrigation.project_paths import REPOSITORY_ROOT
from deeponet_irrigation.training import select_device, write_json


ROOT = REPOSITORY_ROOT
DATA_DIR = ROOT / "data" / "processed" / "arnesano_v2"
RESULTS_DIR = ROOT / "results"
SELECTED_CHECKPOINT = ROOT / "models" / "mlp_baseline_best.pt"
EXPERIMENT_LOG = RESULTS_DIR / "mlp_experiments.json"
FINAL_METRICS = RESULTS_DIR / "final_baseline_metrics.json"
FINAL_PREDICTIONS = RESULTS_DIR / "final_test_predictions.npz"
PRIOR_DIRECT_TEST_PREDICTIONS = RESULTS_DIR / "test_predictions.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the one-time final test evaluation after validation selection"
    )
    parser.add_argument(
        "--final-test",
        action="store_true",
        help="Required acknowledgement that model selection is complete.",
    )
    return parser.parse_args()


def ensure_same_frozen_test(
    current: dict[str, np.ndarray], prior: np.lib.npyio.NpzFile
) -> None:
    for name in ("target", "horizon_hours", "current_index", "target_index"):
        if not np.array_equal(current[name], prior[name]):
            raise ValueError(f"Prior direct predictions use a different test array: {name}")


def print_metrics(label: str, metrics: dict[str, Any]) -> None:
    print(f"\n{label}")
    print(f"{'Horizon':>8} {'N':>7} {'MAE':>10} {'RMSE':>10} {'R2':>10}")
    for horizon, values in [
        ("overall", metrics["overall"]),
        *metrics["by_horizon"].items(),
    ]:
        print(
            f"{horizon:>8} {values['sample_count']:>7d} "
            f"{values['mae']:>10.4f} {values['rmse']:>10.4f} "
            f"{values['r2']:>10.4f}"
        )


def main() -> None:
    args = parse_args()
    if not args.final_test:
        raise SystemExit(
            "Refusing to access test data without --final-test after selection is frozen."
        )
    if FINAL_METRICS.exists() or FINAL_PREDICTIONS.exists():
        raise SystemExit(
            "Final test artifacts already exist; refusing to repeat test evaluation."
        )
    if not SELECTED_CHECKPOINT.exists() or not EXPERIMENT_LOG.exists():
        raise SystemExit(
            "Run `uv run --project python python python/scripts/tune_mlp.py` "
            "from the repository root and freeze validation selection first."
        )
    if not PRIOR_DIRECT_TEST_PREDICTIONS.exists():
        raise SystemExit("Missing the original direct-MLP test prediction artifact.")

    experiment_log = json.loads(EXPERIMENT_LOG.read_text(encoding="utf-8"))
    selected_id = experiment_log.get("selected_experiment_id")
    if not selected_id:
        raise SystemExit("Experiment log does not contain a frozen selection.")
    selected_experiment = next(
        item
        for item in experiment_log["experiments"]
        if item["experiment_id"] == selected_id
    )

    device = select_device()
    dataset = PreparedTemporalDataset(DATA_DIR, "test")
    model, checkpoint = load_mlp_checkpoint(
        SELECTED_CHECKPOINT, dataset.metadata, device
    )
    if checkpoint.get("target_mode") != "residual":
        raise ValueError("The selected checkpoint is not a residual model")
    loader = make_data_loader(dataset, batch_size=1024, shuffle=False)
    started = time.perf_counter()
    predictions = collect_predictions(
        loader, dataset.metadata, device=device, model=model
    )
    evaluation_seconds = time.perf_counter() - started

    with np.load(PRIOR_DIRECT_TEST_PREDICTIONS) as prior:
        ensure_same_frozen_test(predictions, prior)
        direct_prediction = prior["mlp"].copy()
        prior_created_at = datetime.fromtimestamp(
            PRIOR_DIRECT_TEST_PREDICTIONS.stat().st_mtime, tz=UTC
        ).isoformat()

    metrics = {
        "persistence": metrics_by_horizon(
            predictions["target"],
            predictions["persistence"],
            predictions["horizon_hours"],
        ),
        "direct_mlp": metrics_by_horizon(
            predictions["target"], direct_prediction, predictions["horizon_hours"]
        ),
        "selected_residual_mlp": metrics_by_horizon(
            predictions["target"],
            predictions["mlp"],
            predictions["horizon_hours"],
        ),
    }
    change_status = {
        label: error_by_change_status(
            predictions["target"], prediction, predictions["persistence"], tolerance=0.5
        )
        for label, prediction in (
            ("persistence", predictions["persistence"]),
            ("direct_mlp", direct_prediction),
            ("selected_residual_mlp", predictions["mlp"]),
        )
    }

    np.savez_compressed(
        FINAL_PREDICTIONS,
        target=predictions["target"],
        persistence=predictions["persistence"],
        direct_mlp=direct_prediction,
        selected_residual_mlp=predictions["mlp"],
        horizon_hours=predictions["horizon_hours"],
        current_index=predictions["current_index"],
        target_index=predictions["target_index"],
        target_timestamp=np.asarray(dataset.timestamps)[
            predictions["target_index"].astype(np.int64)
        ],
    )
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "final_test_evaluation": True,
        "selection_frozen_before_test": True,
        "selected_experiment_id": selected_id,
        "selected_checkpoint": str(SELECTED_CHECKPOINT),
        "selected_validation_metrics": selected_experiment["validation_metrics"],
        "persistence_validation_metrics": experiment_log[
            "persistence_validation_metrics"
        ],
        "direct_validation_metrics": experiment_log["direct_validation_metrics"],
        "test_metrics": metrics,
        "test_error_by_change_status": change_status,
        "direct_test_prediction_source": {
            "path": str(PRIOR_DIRECT_TEST_PREDICTIONS),
            "note": "Reused from the original pre-task evaluation; direct MLP was not rerun.",
            "artifact_modified_at_utc": prior_created_at,
        },
        "runtime": {
            "torch_version": torch.__version__,
            "device": str(device),
            "gpu_name": (
                torch.cuda.get_device_name(0) if device.type == "cuda" else None
            ),
            "selected_test_inference_seconds": evaluation_seconds,
        },
    }
    write_json(FINAL_METRICS, report)
    for label, values in metrics.items():
        print_metrics(label, values)
    print(f"\nFinal metrics: {FINAL_METRICS}")
    print(f"Final predictions: {FINAL_PREDICTIONS}")


if __name__ == "__main__":
    main()
